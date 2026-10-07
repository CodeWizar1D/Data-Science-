import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const distDir = path.join(__dirname, 'dist')

const htmlPath = path.join(distDir, 'index.html')
const html = fs.readFileSync(htmlPath, 'utf8')

const assetsDir = path.join(distDir, 'assets')
const files = fs.readdirSync(assetsDir)

const cssFile = files.find(f => f.endsWith('.css'))
const jsFile = files.find(f => f.endsWith('.js'))

if (!cssFile || !jsFile) {
  console.error('Could not find CSS or JS file in dist/assets')
  process.exit(1)
}

const css = fs.readFileSync(path.join(assetsDir, cssFile), 'utf8')
const js = fs.readFileSync(path.join(assetsDir, jsFile), 'utf8')

// Crucial: Use function replacers so special patterns like $& in minified JS are NOT evaluated
let bundled = html
bundled = bundled.replace(
  /<link rel="stylesheet"[^>]+>/,
  () => `<style>\n${css}\n</style>`
)
bundled = bundled.replace(
  /<script type="module"[^>]+><\/script>/,
  () => `<script type="module">\n${js}\n</script>`
)

fs.writeFileSync(path.join(distDir, 'index_bundle.html'), bundled, 'utf8')
console.log('Successfully generated clean bundle at dist/index_bundle.html (' + bundled.length + ' bytes)')
