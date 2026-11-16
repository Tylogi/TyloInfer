# 原生服务迁移

`refactor/native-server` 已先合入 `origin/main`，再以 `refactor/arch` 的当前实现完成服务接入。
原生服务位于 `csrc/server/`，入口为 `csrc/apps/server_main.cpp`。

## 结构

- 模型实例直接持有 `tyloi::engine::Engine`，使用它的提交、取消、会话、快照和关闭接口。
- 调度由 Engine 自身管理。原服务的请求数量、输入字节和命令队列预算保留在现有 Scheduler 中。
- `csrc/components/protocol/` 为公共服务和 HTTP/stdio transport 提供共享的解析、输出及运行配置处理。
- CUDA 与 Metal 使用 arch 的 `engine::EngineOptions` 和加载工厂；HIP 保持 arch 的现有实现，尚未接入公共服务工厂。
- Studio 启动原生服务，管理路由使用 `/api`。Python legacy 服务继续保留 `/api/v1`。
- SQLite 使用版本 19 的共享 `schema.sql`，原生下载、模型管理、鉴权、任务与会话代码随服务迁入。
- Windows 打包构建原生服务。macOS 保留 `--prepare-only`，正式 DMG 的 dylib 归集与打包仍待整合。

## 构建入口

无 GPU 的管理服务可按以下方式构建；该配置不提供模型推理：

```sh
cmake -S csrc -B build/native-server \
  -DTYLOINFER_BUILD_SERVER=ON \
  -DTYLOINFER_BUILD_CUDA_ENGINE=OFF \
  -DTYLOINFER_BUILD_HIP_RUNTIME=OFF \
  -DTYLOINFER_BUILD_METAL_RUNTIME=OFF \
  -DTYLOINFER_BUILD_TRANSPORT=OFF \
  -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/native-server --target tyloinfer-server -j
build/native-server/server/tyloinfer-server --data-dir .tyloinfer --port 8090
```

启用 `TYLOINFER_BUILD_CUDA_ENGINE` 或 `TYLOINFER_BUILD_METAL_RUNTIME` 可连接对应的 arch Engine。
服务依赖 SQLite3、libcurl 7.85 及以上版本和 OpenSSL Crypto。
HTTP 接口以 [HTTP API](api/http.md) 为准。

## 验证边界

本次完成无 GPU 原生服务与共享 transport 编译、Studio 生产构建、CUDA 配置及 CUDA/Metal 入口编译检查。
用户要求不再运行或手工修改测试。没有进行 GPU 模型推理或 Windows/macOS 发行验证。
Rust 检查受本机缺少 DBus 开发库限制。
