#include "engine/generation/batch_driver.h"
#include "engine/budget.h"
#include "engine/executor.h"
#include "engine/generation/decode.h"
#include "engine/generation/prefill_budget.h"
#include "engine/weights/cache.h"
#include "engine/weights/workspace.h"

#include <algorithm>
#include <array>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace tyloi::engine {
void reserve_batch(Budget& budget, Device primary_device, const models::BatchGrowth& values,
        std::size_t margin, bool primary) {
    models::BatchGrowth growth;
    for(const auto& [device,bytes]:values) {
        auto found=std::find_if(growth.begin(),growth.end(),[&](const auto& value){return value.first==device;});
        if(found==growth.end())growth.emplace_back(device,bytes);
        else found->second=prefill_budget_add(found->second,bytes);
    }
    if(primary && std::none_of(growth.begin(),growth.end(),[&](const auto& value){return value.first==primary_device;}))
        growth.emplace_back(primary_device,0);
    for(const auto& [device,bytes]:growth)
        if(bytes || (primary && device==primary_device))budget.reserve_kv(device,prefill_budget_add(bytes,margin));
}

void submit_batch(Executor& executor, std::span<const Tensor> outputs,
        std::span<models::BatchState* const> states) {
    std::vector<Tensor> retained(outputs.begin(),outputs.end());
    for(const auto* state:states) {
        auto storage=state->storage();
        retained.insert(retained.end(),storage.begin(),storage.end());
    }
    executor.submit(retained);
}

std::vector<Tensor> prefill_batch(models::Batch& batch, Executor& executor, weights::Cache& weights,
        Budget& budget, Device device, std::vector<models::BatchState*> states, std::vector<Tensor> ids) {
    Capture capture(executor,weights,budget,device);
    auto work=executor.work();
    std::unique_ptr<weights::Loan> loan;
    StepSequence<std::vector<Tensor>> job;
    std::optional<std::vector<Tensor>> result;
    try {
        job=batch.prefill(states,std::move(ids));
        for(;;) {
            StepResult<std::vector<Tensor>> step;
            { auto scope=work->enter(); step=job.next(); }
            const int events=int(bool(step.value))+int(bool(step.layer))+int(bool(step.workspace))+int(bool(step.graph));
            if(events>1 || (step.state==StepState::complete && events))
                throw std::logic_error("invalid batch prefill event");
            if(!step)break;
            if(step.state!=StepState::advanced || result)throw std::logic_error("invalid batch prefill event");
            if(step.layer)work->step(*step.layer);
            else if(step.graph)capture.step(*step.graph);
            else if(step.workspace) {
                const auto request=*step.workspace;
                if(request.bytes) {
                    if(loan || request.output || request.extra)throw std::logic_error("invalid batch workspace acquisition");
                    loan=std::make_unique<weights::Loan>(weights);
                    if(request.tile) {
                        const auto tile=*request.tile;
                        if(!tile.compute || *tile.compute<=0 || tile.tokens<=0 || !tile.row_bytes)
                            throw std::invalid_argument("invalid batch workspace geometry");
                        const auto memory=[&] {
                            const std::array<Device,1> devices{device};
                            const auto snapshot=executor.memory(devices).at(0);
                            return prefill_budget_add(snapshot.allocatable,snapshot.workspace);
                        };
                        const auto available=weights.prefill_budget(true);
                        const auto fixed=prefill_budget_add(request.bytes,available.transfer_bytes);
                        auto selected=select_prefill_chunk(tile.tokens,*tile.compute,
                            prefill_budget_add(memory(),available.lendable_bytes),fixed,tile.row_bytes);
                        if(selected<=0)throw std::runtime_error("insufficient batch prefill workspace before expert loan");
                        const auto rows=std::size_t(selected)>std::numeric_limits<std::size_t>::max()/tile.row_bytes
                            ? std::numeric_limits<std::size_t>::max() : std::size_t(selected)*tile.row_bytes;
                        loan->begin(prefill_budget_add(fixed,rows));
                        selected=select_prefill_chunk(tile.tokens,selected,memory(),
                            prefill_budget_add(request.bytes,loan->transfer_bytes()),tile.row_bytes);
                        if(selected<=0)throw std::runtime_error("insufficient batch prefill workspace after expert loan");
                        *tile.compute=selected;
                    } else loan->begin(request.bytes);
                } else {
                    if(!loan || !request.output || request.tile)throw std::logic_error("invalid batch workspace completion");
                    std::array<Tensor*,2> outputs{request.output,request.extra};
                    loan->finish(std::span<Tensor* const>{outputs.data(),request.extra ? 2U : 1U});
                    loan.reset();
                }
            } else if(step.value)result=std::move(*step.value);
        }
        if(!result)throw std::runtime_error("batch prefill ended without a result");
        if(loan)throw std::logic_error("batch prefill ended while workspace is borrowed");
        submit_batch(executor,*result,states);
        capture.finish(); work->finish(); job={}; return std::move(*result);
    } catch(...) {
        auto failure=std::current_exception();
        try { capture.drain(); } catch(...) {}
        job={}; work.reset(); loan.reset(); std::rethrow_exception(failure);
    }
}

std::unique_ptr<models::Batch::Plan> plan_batch(models::Batch& batch, Executor& executor,
        weights::Cache& weights, Budget& budget, Device device, const std::vector<models::BatchRow>& rows, bool greedy) {
    Capture capture(executor,weights,budget,device);
    std::unique_ptr<models::Batch::Plan> plan;
    StepSequence<std::unique_ptr<models::Batch::Plan>> job;
    try {
        job=batch.plan(rows,greedy);
        for(;;) {
            auto step=job.next();
            if(step.layer || step.workspace || (step.graph && step.value) ||
                (step.state==StepState::complete && (step.graph || step.value)))
                throw std::logic_error("invalid batch graph plan event");
            if(!step)break;
            if(step.state!=StepState::advanced)throw std::logic_error("invalid batch graph plan event");
            if(step.graph) {
                if(plan)throw std::logic_error("batch graph plan continued after its result");
                capture.step(*step.graph);
            } else if(step.value) {
                if(plan || !*step.value)throw std::logic_error("invalid batch graph plan result");
                plan=std::move(*step.value);
            } else if(plan)throw std::logic_error("batch graph plan continued after its result");
        }
        capture.finish(); job={}; return plan;
    } catch(...) {
        auto failure=std::current_exception();
        try { capture.drain(); } catch(...) {}
        job={}; std::rethrow_exception(failure);
    }
}

std::optional<Tensor> capture_batch(models::Batch& batch, Executor& executor,
        weights::Cache& weights, Budget& budget, Device device, const std::vector<models::BatchRow>& rows,
        const Tensor& ids, const Tensor& host_ids, bool greedy, const models::Batch::Plan& plan) {
    Capture capture(executor,weights,budget,device);
    StepSequence<std::optional<Tensor>> job;
    std::optional<std::optional<Tensor>> result;
    try {
        job=batch.capture(rows,ids,host_ids,greedy,plan);
        while (auto step=job.next()) {
            if (step.state!=StepState::advanced || step.layer || step.workspace ||
                (step.graph && step.value) || result)
                throw std::logic_error("invalid batch capture event");
            if (step.graph) capture.step(*step.graph);
            else if (step.value) result.emplace(std::move(*step.value));
        }
        if (!result) throw std::runtime_error("batch capture ended without a result");
        if (*result) {
            std::vector<models::BatchState*> states;
            for (const auto& row:rows) if (row.eligible) states.push_back(row.state);
            submit_batch(executor,std::span<const Tensor>{&**result,1},states);
        }
        capture.finish(); job={}; return std::move(*result);
    } catch (...) {
        auto failure=std::current_exception();
        try { capture.drain(); } catch (...) {}
        job={}; std::rethrow_exception(failure);
    }
}

BatchDriver::BatchDriver(std::unique_ptr<models::Batch> batch, Executor& executor,
        weights::Cache& weights, Budget& budget, Device device, std::size_t workspace_reserve)
    : executor_(executor), weights_(weights), budget_(budget), device_(device),
      workspace_reserve_(workspace_reserve), batch_(std::move(batch)) {
    if(!batch_)throw std::invalid_argument("model does not provide numerical batching");
}

void BatchDriver::initialize() {
    auto scope=executor_.enter();
    reserve_batch(budget_,device_,batch_->initial_growth_bytes(),std::size_t(64)<<20);
    batch_->initialize();
    weights_.finish_exchange();
    executor_.drain();
}

BatchDriver::~BatchDriver() noexcept {
    std::unique_ptr<Executor::Scope> scope;
    try { scope=executor_.enter(); } catch(const std::exception& e) { std::cerr<<"batch_driver_cleanup error="<<e.what()<<'\n'; } catch(...) {}
    try { weights_.drain(); } catch(const std::exception& e) { std::cerr<<"batch_driver_cleanup error="<<e.what()<<'\n'; } catch(...) {}
    try { executor_.drain(); } catch(const std::exception& e) { std::cerr<<"batch_driver_cleanup error="<<e.what()<<'\n'; } catch(...) {}
    try { close(); } catch(const std::exception& e) { std::cerr<<"batch_driver_cleanup error="<<e.what()<<'\n'; } catch(...) {}
}

models::BatchState& BatchDriver::add(std::int64_t prompt_tokens) {
    auto scope=executor_.enter();
    if(states_.empty())weights_.finish_exchange();
    auto state=batch_->make_state(prompt_tokens);
    if(!state)throw std::runtime_error("batch state is unavailable");
    reserve_batch(budget_,device_,state->initial_growth_bytes(),std::size_t(64)<<20);
    states_.push_back(std::move(state));
    states_.back()->initialize();
    return *states_.back();
}

Tensor BatchDriver::prefill(models::BatchState& state, Tensor ids) {
    auto scope=executor_.enter();
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<=0)
        throw std::invalid_argument("batch prompt must be nonempty [1,tokens]");
    if(state.position()!=0)throw std::runtime_error("batch prefill position mismatch");
    const auto count=ids.size(1);
    const auto growth=state.prefill_growth_bytes(count);
    std::size_t primary=0;
    for(const auto& [device,bytes]:growth) {
        if(device==device_)primary=prefill_budget_add(primary,bytes);
        else if(bytes)budget_.reserve_kv(device,prefill_budget_add(bytes,std::size_t(64)<<20));
    }
    const auto chunk=budget_.batch(device_,batch_->prefill_row_bytes(),{0,count,true},primary);
    if(chunk.count!=count)throw std::runtime_error("batch prefill admission cannot preserve the full prompt");
    if(count>1024)weights_.workspace(workspace_reserve_);
    reserve_batch(budget_,device_,growth,std::size_t(256)<<20,true);
    auto outputs=prefill_batch(*batch_,executor_,weights_,budget_,device_,{&state},{std::move(ids)});
    if(outputs.size()!=1)throw std::runtime_error("invalid batch prefill output count");
    return std::move(outputs.front());
}

std::optional<Tensor> BatchDriver::capture(const std::vector<models::BatchRow>& rows,
        const Tensor& ids, const Tensor& host_ids, bool greedy) {
    auto scope=executor_.enter();
    std::unique_ptr<models::Batch::Plan> plan;
    try {
        reserve_batch(budget_,device_,batch_->decode_growth(rows),std::size_t(64)<<20);
        batch_->prepare_decode(rows);
        plan=plan_batch(*batch_,executor_,weights_,budget_,device_,rows,greedy);
        if(!plan)return {};
        if(plan->rebuild) {
            executor_.reclaim(device_);
            reserve_batch(budget_,device_,plan->growth,std::size_t(64)<<20);
        }
        return capture_batch(*batch_,executor_,weights_,budget_,device_,rows,ids,host_ids,greedy,*plan);
    } catch(...) {
        auto failure=std::current_exception();
        try { weights_.drain(); } catch(...) {}
        try { executor_.drain(); } catch(...) {}
        plan.reset();
        std::rethrow_exception(failure);
    }
}

void BatchDriver::close() {
    if(!batch_)return;
    std::exception_ptr failure;
    try { batch_->clear(); } catch(...) { failure=std::current_exception(); }
    for(auto& state:states_)if(state)try { batch_->release(*state); }
        catch(...) { if(!failure)failure=std::current_exception(); }
    batch_.reset(); states_.clear();
    if(failure)std::rethrow_exception(failure);
}
}
