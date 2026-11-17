#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>

class GdnTileScope {
    std::string previous_;
    bool present_;
    static void set(const char* value) {
#ifdef _WIN32
        if(_putenv_s("TYLOINFER_GDN_DECODE_TILES",value)!=0)throw std::runtime_error("cannot set GDN tile option");
#else
        const int result=*value?setenv("TYLOINFER_GDN_DECODE_TILES",value,1):unsetenv("TYLOINFER_GDN_DECODE_TILES");
        if(result!=0)throw std::runtime_error("cannot set GDN tile option");
#endif
    }
public:
    explicit GdnTileScope(int tiles):present_(std::getenv("TYLOINFER_GDN_DECODE_TILES")!=nullptr) {
        if(present_)previous_=std::getenv("TYLOINFER_GDN_DECODE_TILES");
        set(tiles?std::to_string(tiles).c_str():"");
    }
    GdnTileScope(const GdnTileScope&)=delete;
    GdnTileScope& operator=(const GdnTileScope&)=delete;
    ~GdnTileScope(){try{set(present_?previous_.c_str():"");}catch(...) {}}
};
