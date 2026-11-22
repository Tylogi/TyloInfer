#include "text_processor.h"
#include "generation_policy.h"
#include "gguf.h"
#include "mfq_text.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct TokenizerFixture {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("mfq-text-prefix-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    int vocabulary = 0;

    explicit TokenizerFixture(const char* chat_template = "{{ messages[0]['content'] }}") {
        std::vector<std::string> names{"<unk>", "<s>", "</s>"};
        std::vector<std::int32_t> types{TYLOINFER_TEXT_TOKEN_TYPE_UNKNOWN,
            TYLOINFER_TEXT_TOKEN_TYPE_CONTROL, TYLOINFER_TEXT_TOKEN_TYPE_CONTROL};
        for (int value = 0; value < 256; ++value) {
            char name[7];
            std::snprintf(name, sizeof(name), "<0x%02X>", value);
            names.emplace_back(name);
            types.push_back(TYLOINFER_TEXT_TOKEN_TYPE_BYTE);
        }
        for (const auto* name : {"<｜deepseek_image｜>", "<think>", "</think>"}) {
            names.emplace_back(name);
            types.push_back(TYLOINFER_TEXT_TOKEN_TYPE_CONTROL);
        }
        vocabulary = static_cast<int>(names.size());
        std::vector<const char*> tokens;
        for (const auto& name : names) tokens.push_back(name.c_str());
        std::vector<float> scores(names.size());
        auto* metadata = gguf_init_empty();
        require(metadata != nullptr, "cannot create tokenizer fixture");
        gguf_set_val_str(metadata, "general.architecture", "mfq-test");
        gguf_set_val_str(metadata, "tokenizer.ggml.model", "llama");
        gguf_set_arr_str(metadata, "tokenizer.ggml.tokens", tokens.data(), tokens.size());
        gguf_set_arr_data(metadata, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
        gguf_set_arr_data(metadata, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
        gguf_set_val_u32(metadata, "tokenizer.ggml.bos_token_id", 1);
        gguf_set_val_u32(metadata, "tokenizer.ggml.eos_token_id", 2);
        gguf_set_val_bool(metadata, "tokenizer.ggml.add_space_prefix", false);
        gguf_set_val_str(metadata, "tokenizer.chat_template", chat_template);
        const bool written = gguf_write_to_file(metadata, path.string().c_str(), true);
        gguf_free(metadata);
        require(written, "cannot write tokenizer fixture");
    }
    ~TokenizerFixture() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

} // namespace

int main() try {
    using namespace mfq::engine;
    {
        TokenizerFixture fixture(R"JINJA({% set efforts = {"none": 0, "high": 1} %}{{ efforts[reasoning_effort | default('high')] }}:{{ messages[0]['content'] }})JINJA");
        TextProcessor inkling(fixture.path.string(), fixture.vocabulary, "inkling2_loop");
        require(inkling.chat_template_capabilities().thinking, "Inkling did not advertise its thinking toggle");
        InferenceInput input;
        input.chat = true;
        input.chat_input.emplace();
        common_chat_msg message;
        message.role = "user";
        message.content = "abc";
        auto& options = input.chat_input->template_inputs;
        options.messages.push_back(std::move(message));
        options.enable_thinking = false;
        std::string rendered;
        (void)inkling.prepare(input, 1024, &rendered);
        require(rendered == "0:abc", "Inkling did not select zero reasoning effort when thinking was disabled");
        options.chat_template_kwargs["reasoning_effort"] = R"("high")";
        (void)inkling.prepare(input, 1024, &rendered);
        require(rendered == "1:abc", "Inkling overrode an explicit reasoning effort");
    }
    {
        TokenizerFixture fixture(R"JINJA({% set data = {'z': [{'b': '月', 'a': 1}], 'a': messages[0].content} %}{{ data | tojson(sort_keys=true, ensure_ascii=true, separators=(',', ':')) }}|{{ data | tojson }})JINJA");
        TextProcessor inkling(fixture.path.string(), fixture.vocabulary, "inkling2_loop");
        InferenceInput input;
        input.chat = true;
        input.chat_input.emplace();
        common_chat_msg message;
        message.role = "user";
        message.content = "Moon";
        input.chat_input->template_inputs.messages.push_back(std::move(message));
        std::string rendered;
        (void)inkling.prepare(input, 1024, &rendered);
        require(rendered == R"({"a":"Moon","z":[{"a":1,"b":"\u6708"}]}|{"z": [{"b": "月", "a": 1}], "a": "Moon"})",
            "Jinja tojson did not recursively sort keys or mutated the input order");
    }
    {
        const auto templates = common_chat_templates_init(nullptr, R"JINJA({% if tools %}{{ tools | tojson }}{% endif %}
            {{ messages[0].content }}{# <|content_invoke_tool_json|> <|content_model_end_sampling|> #})JINJA");
        common_chat_templates_inputs inputs;
        inputs.messages = {{"user", "Record the Moon"}};
        inputs.tools = {{"record_satellite", "Record a satellite",
            R"({"type":"object","properties":{"body":{"const":"Moon"}},"required":["body"],"additionalProperties":false})"}};
        inputs.enable_thinking = false;
        for (const auto selection : {COMMON_CHAT_TOOL_CHOICE_REQUIRED, COMMON_CHAT_TOOL_CHOICE_AUTO}) {
            inputs.tool_choice = selection;
            const auto format = common_chat_templates_apply(templates.get(), inputs);
            require(!format.grammar.empty() && format.grammar_lazy == (selection == COMMON_CHAT_TOOL_CHOICE_AUTO),
                "Inkling tool grammar does not match the requested tool choice");
            common_chat_parser_params parser(format);
            parser.parser.load(format.parser);
            const auto parsed = common_chat_parse(
                R"(<|message_model|>record_satellite<|content_invoke_tool_json|>{"name":"record_satellite","args":{"body":"Moon"}}<|end_message|>)",
                false, parser);
            require(parsed.tool_calls.size() == 1 && parsed.tool_calls[0].name == "record_satellite" &&
                parsed.tool_calls[0].arguments == R"({"body":"Moon"})", "Inkling tool call was not decoded");
        }
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_NONE;
        inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        const auto format = common_chat_templates_apply(templates.get(), inputs);
        common_chat_parser_params parser(format);
        parser.reasoning_format = inputs.reasoning_format;
        parser.parser.load(format.parser);
        const auto reply = common_chat_parse("<|message_model|><|content_thinking|>Tool confirmed<|end_message|>"
            "<|message_model|><|content_text|>Moon<|end_message|>", false, parser);
        require(format.grammar.empty() && reply.content == "Moon" && reply.reasoning_content == "Tool confirmed",
            "Inkling tool result reply leaked message control tokens");
    }
    TokenizerFixture fixture;
    TextProcessor processor(fixture.path.string(), fixture.vocabulary, "deepseek_v41");
    for (const auto& prompt : std::vector<std::string>{"abc", "<s>abc</s>", std::string(4096, 'a')})
        require(processor.count_tokens(prompt) == processor.tokenizer().tokenize(prompt, false, false).size(),
            "token count differs from raw scoring tokenization");
    require(processor.count_tokens("<s>") > 1, "token count parsed a special token");
    for (const std::string prompt : {std::string(), std::string(262145, 'a')}) {
        bool rejected = false;
        try { (void)processor.count_tokens(prompt); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "token count accepted an unbounded prompt");
    }
    for (int images : {0, 1, 2}) for (bool enabled : {false, true}) {
        InferenceInput input;
        input.chat = true;
        input.sampling.max_tokens = 128000;
        input.sampling.enable_thinking = false;
        input.prefix_cache_enabled = enabled;
        input.chat_input.emplace();
        std::string prompt = "a";
        for (int i = 0; i < images; ++i) prompt += "<｜deepseek_image｜>b";
        prompt += "</think>";
        input.chat_input->preformatted_prompt = prompt;
        input.chat_input->template_inputs.add_generation_prompt = true;
        if (images) {
            input.media.emplace();
            input.media->processor = MfqMultimodalProcessor::deepseek_v41;
            input.media->processor_name = "deepseek_v41";
            input.media->pixel_shape = {images, 18, 588};
            input.media->pixel_values.resize(images * 18 * 588);
            input.media->vision_grid_shape = {images, 4};
            for (int i = 0; i < images; ++i)
                input.media->vision_grid.insert(input.media->vision_grid.end(), {3, 6, 1, 2});
        }
        const auto result = processor.prepare(std::move(input), 1024);
        const auto plan = plan_generation(result.prompt, fixture.vocabulary, 1024, result.sampling.max_tokens);
        require(plan.generation_tokens == 1024 - static_cast<std::int32_t>(result.prompt.size()),
            "output budget did not fit the remaining context after tokenization");
        require(result.cache_plan.stable_prefix_tokens == (enabled ? result.prompt.size() - 1 : 0),
            "image expansion moved the stable prefix away from its token boundary");
        if (images) {
            require(result.vision && result.vision->image_bounds.size() == static_cast<std::size_t>(images * 4),
                "image expansion lost image bounds");
            require(result.cache_plan.stable_prefix_tokens == 0 ||
                    result.cache_plan.stable_prefix_tokens >= static_cast<std::size_t>(result.vision->image_bounds.back()),
                "stable prefix does not include expanded image spans");
        }
    }
    InferenceInput chat;
    chat.chat = true;
    chat.chat_input.emplace();
    chat.chat_input->preformatted_prompt = "abc";
    chat.sampling.max_tokens = 128000;
    const auto prompt_tokens = processor.prepare(chat, 128000).prompt.size();
    const auto full = processor.prepare(chat, prompt_tokens);
    require(plan_generation(full.prompt, fixture.vocabulary, prompt_tokens, full.sampling.max_tokens)
            .generation_tokens == 0,
        "a prompt filling the context must allow zero output tokens");
    bool over_context_rejected = false;
    try { (void)processor.prepare(chat, prompt_tokens - 1); }
    catch (const InferenceInputError& error) {
        over_context_rejected = error.code == InferenceInputErrorCode::ContextLength;
    }
    require(over_context_rejected, "an oversized prompt must return a context length error");

    DecisionText decision;
    decision.reads={{"abc",{"A","B"}},{"xyz",{"0","1"}}};
    decision.cache_salt="isolated";decision.prefix_cache=false;decision.full_vocabulary=false;
    const auto prepared=processor.prepare_decisions(decision,1024);
    require(prepared.reads.size()==2 && prepared.cache_salt=="isolated" &&
        !prepared.prefix_cache && !prepared.full_vocabulary,"decision read flags changed");
    for(const auto& read:prepared.reads)
        require(!read.prompt.empty() && read.labels.size()==2 && read.labels[0]!=read.labels[1],
            "decision labels lost their reply-boundary token alignment");
    for(const auto& labels:std::vector<std::vector<std::string>>{{"A","A"},{"several"},{"</s>"},{""},{}}) {
        auto invalid=decision;invalid.reads[0].labels=labels;
        bool rejected=false;
        try {processor.prepare_decisions(invalid,1024);}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"decision accepted a duplicate, special or multi-token label");
    }
    bool context_rejected=false;
    try{processor.prepare_decisions(decision,1);}catch(const InferenceInputError&){context_rejected=true;}
    require(context_rejected,"decision preparation accepted a prompt beyond the loaded context");
    std::cout << "MFQ text processor media-prefix and decision boundaries passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
