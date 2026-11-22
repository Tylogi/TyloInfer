#pragma once
#include "mfq_cuda_paged_kv.h"

#include "tests/numeric_assertions.h"

void kv_quantization_case() {
    using namespace mfq::cuda;
    for (int bits:{2,3,4,6,8}) {
        constexpr int latent=512,rotary=64,batch=3,capacity=9;
        const auto raw_latent=data({batch,capacity,latent},27,.5f).to(tb::kFloat16);
        const auto raw_rotary=data({batch,capacity,rotary},33,.5f).to(tb::kFloat16);
        const auto x=cuda_kv_encode(raw_latent,bits,false);
        const auto y=cuda_kv_encode(raw_rotary,bits,false);
        const auto combined=cuda_kv_encode_mla(tb::cat({raw_latent,raw_rotary},-1),latent,rotary,bits);
        const auto separate=tb::cat({x,y},-1).to(tb::kCPU),packed=combined.to(tb::kCPU);
        if (std::memcmp(separate.data_ptr(),packed.data_ptr(),packed.nbytes())!=0)
            throw std::runtime_error("Compact MLA quantization changed the packed bytes");
        auto expected=tb::cat({cuda_kv_decode(x,latent,bits,false),cuda_kv_decode(y,rotary,bits,false)},-1).to(tb::kCPU);
        auto lengths=tb::tensor(std::vector<int32_t>{2,0,capacity}).to(tb::kCUDA);
        auto decoded=cuda_kv_decode_mla(combined,latent,rotary,bits,lengths).to(tb::kCPU);
        for (int row:{0,2}) {
            const auto count=row==0?2:capacity;
            const auto a=expected.select(0,row).narrow(0,0,count);
            const auto b=decoded.select(0,row).narrow(0,0,count);
            if (std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes())!=0)
                throw std::runtime_error("Compact MLA dequantization changed a populated row");
        }
    }
    for(double bits:{2.,2.5,3.,3.5,4.,6.,8.})for(int d:{8,64,128,256,384,512}) {
        constexpr int b=2,h=6,kh=2,t=3,length=513;
        KvQuantization quant{bits};
        auto q=data({b,h,t,d},23,.5f).to(tb::kFloat32);
        auto k=data({b,kh,length,d},27,.5f).to(tb::kFloat32);
        auto v=data({b,kh,length,d},33,.5f).to(tb::kFloat32);
        auto pk=cuda_kv_encode(k,quant.key_bits(),false),pv=cuda_kv_encode(v,quant.value_bits(),true);
        auto dk=cuda_kv_decode(pk,d,quant.key_bits(),false,tb::kFloat32);
        auto dv=cuda_kv_decode(pv,d,quant.value_bits(),true,tb::kFloat32);
        if(d>=64 && pk.nbytes()+pv.nbytes()>=k.numel()*4)throw std::runtime_error("TurboQuant did not reduce FP16 KV bytes");
        auto zeros=cuda_kv_decode(cuda_kv_encode(tb::zeros({2,d},k.options()),quant.key_bits(),false),d,quant.key_bits(),false,tb::kFloat32).to(tb::kCPU);
        for(int i=0;i<2*d;++i)if(zeros.data_ptr<float>()[i]!=0)throw std::runtime_error("TurboQuant zero norm changed");
        auto original=k.to(tb::kCPU),decoded=dk.to(tb::kCPU);
        double error=0,energy=0;
        for(int64_t i=0;i<k.numel();++i){double x=original.data_ptr<float>()[i],delta=
            mfq::test::checked_absolute_error(double(decoded.data_ptr<float>()[i]),x);error+=delta*delta;energy+=x*x;}
        const double relative=std::sqrt(error/energy);
        if(!std::isfinite(relative) || relative>(bits<3?.5:bits<4?.3:bits<6?.2:.08))
            throw std::runtime_error("TurboQuant reconstruction error too large");
        auto pos=tb::tensor(std::vector<int64_t>{510,511,512,39,41,44}).reshape({b,t}).to(tb::kCUDA);
        auto lengths=tb::tensor(std::vector<int64_t>{513,43}).to(tb::kCUDA);
        auto mask=tb::ones({b,length},q.options());mask.narrow(1,2,3).fill_(0);
        for(bool sparse:{false,true}) {
            constexpr int n=9;
            std::vector<int32_t> ids(b*t*n);
            for(int row=0;row<b*t;++row)for(int j=0;j<n;++j)ids[row*n+j]=j==0?-1:j==8?600:(row*11+j*7)%length;
            auto indices=tb::empty({b,t,n},tb::TensorOptions().device(tb::kCPU).dtype(tb::kInt32));
            std::copy(ids.begin(),ids.end(),indices.data_ptr<int32_t>());
            auto selected=sparse?indices.to(tb::kCUDA):Tensor{};
            auto output=cuda_kv_attention(q,pk,pv,quant,selected,pos,0,{},length,lengths,mask).to(tb::kCPU);
            auto qc=q.to(tb::kCPU),kc=dk.to(tb::kCPU),vc=dv.to(tb::kCPU);
            auto pc=pos.to(tb::kCPU);const int bounds[2]={513,43};
            double maximum_error=0;
            for(int batch=0;batch<b;++batch)for(int token=0;token<t;++token)for(int head=0;head<h;++head) {
                std::vector<double> scores,weights;std::vector<int> chosen;double top=-INFINITY;
                const int visible=std::min(int(pc.data_ptr<int64_t>()[batch*t+token])+1,bounds[batch]);
                for(int j=0;j<(sparse?n:length);++j) {
                    int id=sparse?ids[(batch*t+token)*n+j]:j;
                    if(id<0||id>=visible||(id>=2&&id<5))continue;
                    double dot=0;
                    for(int dim=0;dim<d;++dim)dot+=double(qc.data_ptr<float>()[((batch*h+head)*t+token)*d+dim])*kc.data_ptr<float>()[((batch*kh+head/(h/kh))*length+id)*d+dim];
                    scores.push_back(dot/std::sqrt(double(d)));chosen.push_back(id);top=std::max(top,scores.back());
                }
                double mass=0;for(double score:scores){weights.push_back(std::exp(score-top));mass+=weights.back();}
                for(int dim=0;dim<d;++dim) {
                    double expected=0;
                    for(size_t j=0;j<chosen.size();++j)expected+=weights[j]*vc.data_ptr<float>()[((batch*kh+head/(h/kh))*length+chosen[j])*d+dim];
                    if(mass)expected/=mass;
                    maximum_error=std::max(maximum_error,mfq::test::checked_absolute_error(
                        expected, output.data_ptr<float>()[((batch*t+token)*h+head)*d+dim]));
                }
            }
            if(maximum_error>3e-5)throw std::runtime_error("TurboQuant compressed attention differs from decoded CPU oracle: "+std::to_string(maximum_error));
        }
        // Noncontiguous physical pages, separate key/value widths, both batch rows.
        constexpr int page_size=16,chunk_pages=64,logical_pages=(length+page_size-1)/page_size;
        auto table=tb::empty({b,logical_pages},tb::TensorOptions().device(tb::kCPU).dtype(tb::kInt32));
        for(int row=0;row<b;++row)for(int p=0;p<logical_pages;++p)table.data_ptr<int32_t>()[row*logical_pages+p]=2*p+(1-row);
        std::vector<Tensor> key_chunks,value_chunks;std::vector<int64_t> kp,vp;
        for(int c=0;c<2;++c) {
            key_chunks.push_back(tb::zeros({chunk_pages,kh,page_size,pk.size(-1)},pk.options()));
            value_chunks.push_back(tb::zeros({chunk_pages,kh,page_size,pv.size(-1)},pv.options()));
            kp.push_back(reinterpret_cast<int64_t>(key_chunks.back().data_ptr()));vp.push_back(reinterpret_cast<int64_t>(value_chunks.back().data_ptr()));
        }
        QuantizedKvPages pages{tb::tensor(kp).to(tb::kCUDA),tb::tensor(vp).to(tb::kCUDA),table.to(tb::kCUDA),page_size,chunk_pages,kh};
        cuda_kv_write_paged(pages,pk,pv,tb::arange(length,pos.options()));
        if (pk.size(-1)!=pv.size(-1)) {
            auto keys=tb::empty_like(pk),values=tb::empty_like(pv);
            paged_kv_cache_gather_cuda(pages.keys,pages.values,pages.table,keys,values,page_size,chunk_pages);
            const auto equal_bytes=[](const Tensor& first,const Tensor& second) {
                const auto a=first.to(tb::kCPU),b=second.to(tb::kCPU);
                return a.nbytes()==b.nbytes() && std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes())==0;
            };
            if (!equal_bytes(keys,pk) || !equal_bytes(values,pv))
                throw std::runtime_error("Paged prefix changed packed TurboQuant rows");
        }
        auto resident=cuda_kv_attention(q,pk,pv,quant,{},pos,0,{},length,lengths,mask);
        auto paged=cuda_kv_attention(q,{},{},quant,{},pos,0,pages,length,lengths,mask);
        equal(paged,resident,"TurboQuant paged attention",1e-6,1e-6);
        if(bits==4&&d==128) {
            // Long prefill uses one partial per row; decode splits the same rows.
            auto prompt=data({b,h,192,d},41,.5f).to(tb::kFloat32);
            auto whole=cuda_kv_attention(prompt,pk,pv,quant,{},{},300);
            for(int token:{0,95,191}) {
                auto single=cuda_kv_attention(prompt.narrow(2,token,1),pk,pv,quant,{},{},300+token);
                equal(whole.narrow(1,token,1),single,"TurboQuant prefill versus decode",3e-5,3e-5);
            }
        }
    }
    int device=0,major=0;
    TYLOINFER_CUDA_CHECK(cudaGetDevice(&device));
    TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device));
    if(major>=7) {
        constexpr int d=256,kv_heads=2;
        for(const int ratio:{4,6,8})for(const int prefix:{0,163,384}) {
            constexpr int tokens=128;
            const int length=prefix+tokens,heads=kv_heads*ratio;
            const auto q=data({1,heads,tokens,d},23,.5f).to(tb::kFloat32);
            const auto k=cuda_kv_encode(data({1,kv_heads,length,d},27,.5f),8,false);
            const auto v=cuda_kv_encode(data({1,kv_heads,length,d},33,.5f),8,true);
            const auto fast=mfq_attention_mma256_prefix_cuda(q,cuda_kv_decode(k,d,8,false),
                cuda_kv_decode(v,d,8,true),1.0/std::sqrt(double(d)));
            if(!fast.defined())continue;
            const auto reference=cuda_kv_attention(q,k,v,KvQuantization{8},{},{},prefix,{},length);
            equal(fast,reference,"TurboQuant MMA causal prefill and append");
        }
    }
    std::cout<<"TurboQuant 7 levels, 6 widths: size, reconstruction, dense/sparse CPU oracle, paged KV and MMA prefill PASS\n";
}
