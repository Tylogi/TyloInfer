#include "decision.h"
#include "decision_protocol.h"
#include "tyloinfer/decision_model.h"

#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>

using namespace mfq::engine;
using namespace mfq::transport_detail;

template<class F> void rejects(F function) {
    bool rejected=false;
    try{function();}catch(const std::exception&){rejected=true;}
    assert(rejected);
}

int main() {
    const std::array<int64_t,2> labels{7,3};
    const std::array<double,2> logits{std::log(2.),0};
    auto read=normalize_decision(labels,logits,std::log(10.),7);
    assert(std::abs(read.probabilities[0]-2./3)<1e-12);
    assert(std::abs(*read.label_mass-.3)<1e-12 && std::abs(read.confidence-.2)<1e-12);
    assert(read.argmax_is_label && *read.argmax_is_label);
    const std::array<double,2> shifted{10000+std::log(2.),10000};
    auto high=normalize_decision(labels,shifted);
    assert(std::abs(high.probabilities[0]-2./3)<1e-11);
    assert(std::abs(std::accumulate(high.probabilities.begin(),high.probabilities.end(),0.)-1)<1e-11);
    assert(!high.label_mass && !high.argmax_is_label);
    const std::array<double,2> extreme{1e100,1e100};
    assert(normalize_decision(labels,extreme).probabilities==std::vector<double>({.5,.5}));
    const std::array<int64_t,1> one{7};const std::array<double,1> one_logit{0};
    assert(normalize_decision(one,one_logit).probabilities[0]==1);
    rejects([&]{normalize_decision({},{});});
    rejects([&]{normalize_decision(one,logits);});
    const std::array<int64_t,2> duplicate{7,7};
    rejects([&]{normalize_decision(duplicate,logits);});
    const std::array<double,2> bad{0,std::numeric_limits<double>::quiet_NaN()};
    rejects([&]{normalize_decision(labels,bad);});
    rejects([&]{normalize_decision(labels,logits,0.);});

    const auto body=json::parse(R"({"model":"sample","state":{"value":"blue"},"questions":{
        "bucket":{"type":"choice","instructions":"Choose a color.","criteria":{"blue":null,"green":"another color"}},
        "matches":{"type":"noul","criteria":{"true":"the value is blue"}},
        "strength":{"type":"score","criteria":["low","high"]}}})");
    auto request=parse_decision_request(body);
    assert(request.questions.size()==3 && request.questions[0].id=="bucket");
    assert(request.input.reads[0].labels==std::vector<std::string>({"A","B"}));
    assert(request.input.reads[1].labels==std::vector<std::string>({"yes","no"}));
    assert(request.input.reads[2].labels==std::vector<std::string>({"0","1"}));
    assert(request.input.reads[0].prompt.find("State:\n{\"value\":\"blue\"}")!=std::string::npos);
    DecisionResult result;
    for(int i=0;i<3;++i){auto value=read;value.prompt_tokens=19;value.cached_tokens=7;value.prefill_ms=2;
        result.reads.push_back(value);}
    const auto rendered=decision_result_json(request,result,"fallback");
    assert(rendered["answers"]["bucket"]["choice"]=="blue");
    assert(std::abs(rendered["answers"]["matches"]["noul"].get<double>()-2./3)<1e-12);
    assert(std::abs(rendered["answers"]["strength"]["score"].get<double>()-1./3)<1e-12);
    assert(rendered["answers"]["strength"]["legend"]["1"]=="high");
    assert(rendered["usage"]["input_tokens"]==57 && rendered["usage"]["output_tokens"]==0);
    assert(rendered["usage"]["input_tokens_details"]["cached_tokens"]==21);
    assert(rendered["diagnostics"]["bucket"]["prefill_ms"]==2);
    assert(rendered["diagnostics"]["bucket"]["prefill_tps"]==6000);
    assert(rendered["model"]=="sample");
    for(const auto& patch:{json{{"unknown",true}},json{{"joint",true}},json{{"joint",1}},
            json{{"questions",json::array()}},json{{"chat_template_kwargs",{{"chat_template","override"}}}}}) {
        auto invalid=body;invalid.update(patch);rejects([&]{parse_decision_request(invalid);});
    }
    for(const auto& criteria:{json::object(),json::array({"x"}),json(true),json(nullptr)}) {
        auto invalid=body;invalid["questions"]["bucket"]["criteria"]=criteria;
        rejects([&]{parse_decision_request(invalid);});
    }
    auto options=json::object();for(int i=0;i<255;++i)options[std::to_string(i)]=nullptr;
    auto invalid=body;invalid["questions"]["bucket"]["criteria"]=options;
    assert(parse_decision_request(invalid).input.reads.front().labels.size()==255);
    invalid["questions"]["bucket"]["criteria"]["overflow"]=nullptr;
    rejects([&]{parse_decision_request(invalid);});
    auto conditional=body;conditional["full_vocabulary"]=false;
    rejects([&]{parse_decision_request(conditional);});
    assert(!parse_decision_request(conditional,true).input.full_vocabulary);
    result.reads.pop_back();rejects([&]{decision_result_json(request,result,"sample");});

    const auto ordered = parse_decision_request(nlohmann::ordered_json::parse(
        R"({"state":{"z":"inside, colon:","a":[1,2]},"questions":{
            "z":{"type":"choice","criteria":{"green":null,"blue":null}},
            "a":{"type":"noul"}}})"));
    assert(ordered.questions.front().id == "z");
    assert(ordered.questions.front().options == std::vector<std::string>({"green", "blue"}));
    assert(ordered.input.reads.front().state == R"({"z": "inside, colon:", "a": [1, 2]})");

    json spec{{"format_version",1},{"format_id","d3-code-readout-v1"},{"attention_mode","prefix_bidir"},
        {"temperature",0.75},{"validated_length",32768},{"max_length",65536}};
    for(int i=0;i<255;++i){spec["codes"].push_back("code"+std::to_string(i));spec["token_ids"].push_back(i);}
    const auto trained=mfq::parse_decision_model(spec.dump());
    assert(trained.attention==mfq::DecisionAttention::prefix_bidirectional && trained.max_tokens==32768);
    assert(trained.temperature==0.75 && trained.codes.size()==255);
    auto calibrated = spec;
    calibrated["prompt_layout"] = "header-v1";
    calibrated["temperature_by_questions"] = {{"max_questions", {1, 4}}, {"temperature", {1.05, 1.1, 1.2}}};
    const auto header = mfq::parse_decision_model(calibrated.dump());
    assert(header.question_header);
    assert(header.temperature_for("choice", 1) == 1.05);
    assert(header.temperature_for("noul", 4) == 1.1);
    assert(header.temperature_for("score", 5) == 1.2);
    calibrated["temperature_by_questions"]["max_questions"] = {4, 1};
    rejects([&] { mfq::parse_decision_model(calibrated.dump()); });
    for(const auto& patch:{json{{"attention_mode","unknown"}},json{{"temperature",0}},
        json{{"pooling","mean"}},json{{"max_length",-1}},json{{"format_version",2}},
        json{{"prompt","other"}},json{{"codes",json::array({"A"})}}}) {
        auto bad=spec;bad.update(patch);rejects([&]{mfq::parse_decision_model(bad.dump());});
    }
    auto duplicate_spec=spec;duplicate_spec["token_ids"][254]=0;
    rejects([&]{mfq::parse_decision_model(duplicate_spec.dump());});
}
