#pragma once

// One declaration per route handler, grouped by resource; routes.cpp pairs these with
// method, path, scope and body limit. The implementation files keep their file-local helpers.

#include "routes.h"

namespace mfq::server {

// api/auth.cpp
void auth_policy_get(const Services&, const httplib::Request&, httplib::Response&);
void auth_policy_put(const Services&, const httplib::Request&, httplib::Response&);
void auth_keys(const Services&, const httplib::Request&, httplib::Response&);
void auth_key_create(const Services&, const httplib::Request&, httplib::Response&);
void auth_key_revoke(const Services&, const httplib::Request&, httplib::Response&);
void auth_key_rotate(const Services&, const httplib::Request&, httplib::Response&);

// api/models.cpp (catalog, directories, load/unload, analysis, deletion)
void models_list(const Services&, const httplib::Request&, httplib::Response&);
void models_browse(const Services&, const httplib::Request&, httplib::Response&);
void models_open(const Services&, const httplib::Request&, httplib::Response&);
void models_register(const Services&, const httplib::Request&, httplib::Response&);
void model_directory(const Services&, const httplib::Request&, httplib::Response&);
void models_load(const Services&, const httplib::Request&, httplib::Response&);
void models_unload(const Services&, const httplib::Request&, httplib::Response&);

// api/artifacts.cpp
void model_analysis(const Services&, const httplib::Request&, httplib::Response&);
void model_cache_profile(const Services&, const httplib::Request&, httplib::Response&);
void model_deletion(const Services&, const httplib::Request&, httplib::Response&);
void model_delete(const Services&, const httplib::Request&, httplib::Response&);

// api/runtime.cpp
void runtime_instances(const Services&, const httplib::Request&, httplib::Response&);
void runtime_instance(const Services&, const httplib::Request&, httplib::Response&);
void runtime_resources(const Services&, const httplib::Request&, httplib::Response&);
void runtime_aliases_get(const Services&, const httplib::Request&, httplib::Response&);
void runtime_aliases_put(const Services&, const httplib::Request&, httplib::Response&);

// api/runtime_config.cpp
void context_policy_get(const Services&, const httplib::Request&, httplib::Response&);
void context_policy_put(const Services&, const httplib::Request&, httplib::Response&);
void inference_policy_get(const Services&, const httplib::Request&, httplib::Response&);
void inference_policy_put(const Services&, const httplib::Request&, httplib::Response&);
void memory_policy_get(const Services&, const httplib::Request&, httplib::Response&);
void memory_policy_put(const Services&, const httplib::Request&, httplib::Response&);
void profiles_create(const Services&, const httplib::Request&, httplib::Response&);
void profiles_list(const Services&, const httplib::Request&, httplib::Response&);
void profile_get(const Services&, const httplib::Request&, httplib::Response&);
void profile_update(const Services&, const httplib::Request&, httplib::Response&);
void profile_delete(const Services&, const httplib::Request&, httplib::Response&);
void profile_load(const Services&, const httplib::Request&, httplib::Response&);
void runtime_instance_patch(const Services&, const httplib::Request&, httplib::Response&);

// api/runtime_cache.cpp
void cache_entries(const Services&, const httplib::Request&, httplib::Response&);
void cache_block_text(const Services&, const httplib::Request&, httplib::Response&);
void cache_purge(const Services&, const httplib::Request&, httplib::Response&);
void cache_clear(const Services&, const httplib::Request&, httplib::Response&);
void cache_trim(const Services&, const httplib::Request&, httplib::Response&);

// api/runtime_control.cpp
void runtime_context(const Services&, const httplib::Request&, httplib::Response&);
void voice_status(const Services&, const httplib::Request&, httplib::Response&);
void voice_install(const Services&, const httplib::Request&, httplib::Response&);
void voice_activate(const Services&, const httplib::Request&, httplib::Response&);
void runtime_yarn(const Services&, const httplib::Request&, httplib::Response&);
void qsa_kv_get(const Services&, const httplib::Request&, httplib::Response&);
void qsa_kv_put(const Services&, const httplib::Request&, httplib::Response&);

// api/jobs.cpp
void jobs(const Services&, const httplib::Request&, httplib::Response&);
void jobs_clear(const Services&, const httplib::Request&, httplib::Response&);
void job_delete(const Services&, const httplib::Request&, httplib::Response&);
void job_get(const Services&, const httplib::Request&, httplib::Response&);
void job_cancel(const Services&, const httplib::Request&, httplib::Response&);
void job_events(const Services&, const httplib::Request&, httplib::Response&);
void job_events_stream(const Services&, const httplib::Request&, httplib::Response&);

// api/hub.cpp
void hub_directory_get(const Services&, const httplib::Request&, httplib::Response&);
void hub_directory_put(const Services&, const httplib::Request&, httplib::Response&);
void hub_search(const Services&, const httplib::Request&, httplib::Response&);
void hub_official(const Services&, const httplib::Request&, httplib::Response&);
void hub_readme(const Services&, const httplib::Request&, httplib::Response&);
void hub_repository(const Services&, const httplib::Request&, httplib::Response&);
void hub_resolve(const Services&, const httplib::Request&, httplib::Response&);
void hub_reuse(const Services&, const httplib::Request&, httplib::Response&);
void hub_probe(const Services&, const httplib::Request&, httplib::Response&);
void jobs_submit(const Services&, const httplib::Request&, httplib::Response&);
void job_retry(const Services&, const httplib::Request&, httplib::Response&);

// api/anthropic.cpp
void anthropic_count_tokens(const Services&, const httplib::Request&, httplib::Response&);
void anthropic_messages(const Services&, const httplib::Request&, httplib::Response&);

// api/inference.cpp
void compat_root(const Services&, const httplib::Request&, httplib::Response&);
void compat_models(const Services&, const httplib::Request&, httplib::Response&);
void compat_model(const Services&, const httplib::Request&, httplib::Response&);
void inference_cancel(const Services&, const httplib::Request&, httplib::Response&);
void chat_completions(const Services&, const httplib::Request&, httplib::Response&);

// api/sessions.cpp
void sessions_create(const Services&, const httplib::Request&, httplib::Response&);
void sessions_list(const Services&, const httplib::Request&, httplib::Response&);
void sessions_import(const Services&, const httplib::Request&, httplib::Response&);
void session_get(const Services&, const httplib::Request&, httplib::Response&);
void session_update(const Services&, const httplib::Request&, httplib::Response&);
void session_delete(const Services&, const httplib::Request&, httplib::Response&);
void session_messages(const Services&, const httplib::Request&, httplib::Response&);
void session_message_add(const Services&, const httplib::Request&, httplib::Response&);
void session_fork(const Services&, const httplib::Request&, httplib::Response&);
void session_rewind(const Services&, const httplib::Request&, httplib::Response&);
void session_export(const Services&, const httplib::Request&, httplib::Response&);
void session_responses(const Services&, const httplib::Request&, httplib::Response&);
void session_cancel(const Services&, const httplib::Request&, httplib::Response&);
void session_respond(const Services&, const httplib::Request&, httplib::Response&);
void presets_list(const Services&, const httplib::Request&, httplib::Response&);
void presets_create(const Services&, const httplib::Request&, httplib::Response&);
void preset_update(const Services&, const httplib::Request&, httplib::Response&);
void preset_delete(const Services&, const httplib::Request&, httplib::Response&);

// api/media.cpp
void media_upload(const Services&, const httplib::Request&, httplib::Response&);
void media_get(const Services&, const httplib::Request&, httplib::Response&);

} // namespace mfq::server
