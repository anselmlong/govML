import { execFileSync } from 'node:child_process';
import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, extname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(fileURLToPath(new URL('../package.json', import.meta.url)));
const frontend = join(root, 'frontend');
const frontendDist = join(frontend, 'dist');
const out = join(root, 'dist');

execFileSync('npm', ['--prefix', frontend, 'ci'], { stdio: 'inherit' });
execFileSync('npm', ['--prefix', frontend, 'run', 'build'], { stdio: 'inherit' });

rmSync(out, { recursive: true, force: true });
mkdirSync(join(out, 'server'), { recursive: true });
mkdirSync(join(out, '.openai'), { recursive: true });

const contentTypes = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2'
};

function walk(dir) {
  const entries = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) entries.push(...walk(full));
    else entries.push(full);
  }
  return entries;
}

const assets = {};
for (const file of walk(frontendDist)) {
  const rel = '/' + relative(frontendDist, file).replaceAll('\\', '/');
  const ext = extname(file).toLowerCase();
  assets[rel] = {
    type: contentTypes[ext] || 'application/octet-stream',
    body: readFileSync(file).toString('base64')
  };
}

const server = `const assets = ${JSON.stringify(assets)};

function decodeBase64(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export default {
  async fetch(request) {
    const url = new URL(request.url);
    let path = decodeURIComponent(url.pathname);
    if (path === '/') path = '/index.html';
    let asset = assets[path];
    if (!asset && !path.includes('.')) asset = assets['/index.html'];
    if (!asset) return new Response('Not found', { status: 404 });
    const headers = new Headers({
      'content-type': asset.type,
      'cache-control': path === '/index.html' ? 'no-cache' : 'public, max-age=31536000, immutable'
    });
    return new Response(decodeBase64(asset.body), { headers });
  }
};
`;

writeFileSync(join(out, 'server', 'index.js'), server);
writeFileSync(join(out, '.openai', 'hosting.json'), readFileSync(join(root, '.openai', 'hosting.json')));

