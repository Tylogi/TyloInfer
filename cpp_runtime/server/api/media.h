#pragma once

#include "media/media.h"
#include "httplib.h"

namespace mfq::server {

void register_media(httplib::Server& http, Media& media);

} // namespace mfq::server
