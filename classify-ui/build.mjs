import * as esbuild from 'esbuild';
import { copyFileSync, mkdirSync } from 'fs';

const watch = process.argv.includes('--watch');

mkdirSync('dist', { recursive: true });

const buildOptions = {
  entryPoints: ['src/main.jsx'],
  bundle: true,
  format: 'esm',
  outfile: 'dist/app.js',
  jsx: 'automatic',
  target: ['es2020'],
  loader: {
    '.js': 'jsx',
  },
  define: {
    'process.env.NODE_ENV': JSON.stringify(watch ? 'development' : 'production'),
  },
  minify: !watch,
  sourcemap: watch,
};

if (watch) {
  const ctx = await esbuild.context(buildOptions);
  await ctx.watch();
  console.log('Watching classify-ui source for changes...');
} else {
  await esbuild.build(buildOptions);
  copyFileSync('index.html', 'dist/index.html');
  copyFileSync('style.css', 'dist/style.css');
  console.log('Built classify-ui/dist/');
}
