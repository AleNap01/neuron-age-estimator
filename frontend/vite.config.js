import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
  },
  build: {
    rollupOptions: {
      output: {
        // three.js/fiber è pesante e condiviso da landing + dashboard: in un
        // chunk separato il browser lo scarica una volta sola e lo cachea,
        // invece di duplicarlo o farlo ribloccare il chunk principale.
        manualChunks: {
          "three-vendor": ["three", "@react-three/fiber", "@react-three/drei"],
          "motion-vendor": ["framer-motion"],
        },
      },
    },
  },
});
