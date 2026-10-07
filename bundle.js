import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const distDir = path.join(__dirname, 'dist')

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

// Cleanly construct single-file HTML without regex/replace ambiguities
const outputHtml = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>DiabPredict — Know Your Diabetes Risk</title>
    <meta name="description" content="A premium AI-powered diabetes risk assessment platform. Private, instant, and built for you." />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet" />
    <style>
${css}
    </style>
  </head>
  <body>
    <div id="root"></div>
    <script type="module">
${js}
    </script>
  </body>
</html>`

fs.writeFileSync(path.join(distDir, 'index_bundle.html'), outputHtml, 'utf8')
console.log('Successfully generated clean bundle at dist/index_bundle.html (' + outputHtml.length + ' bytes)')
