// Vite resolves these imports; tsc only needs their types.
declare module '*.css' { const css: string; export default css; }
declare module '*.css?inline' { const css: string; export default css; }
declare module '*.svg' { const url: string; export default url; }
declare module '*.svg?raw' { const svg: string; export default svg; }
