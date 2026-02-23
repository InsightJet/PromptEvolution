import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../dist/pipeline',
    emptyOutDir: true,
    rollupOptions: {
      input: 'src/main.jsx',
      output: {
        entryFileNames: 'pipeline-builder.js',
        assetFileNames: 'pipeline-builder.[ext]',
        // Single bundle, no code splitting
        manualChunks: undefined,
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
});
