const fs = require('fs-extra');
const path = require('path');
const del = require('del');
const prettier = require('prettier');

const src = path.resolve(__dirname, '../lang');
const dest = path.resolve(__dirname, '../build/messages');

async function main() {
  const files = await getFiles(src);
  await cleanDest(dest);
  await formatFiles(src, dest, files);
}

async function getFiles(srcDir) {
  try {
    const files = await fs.readdir(srcDir);
    return files;
  } catch (e) {
    console.error(`Error reading files from ${srcDir}`, e);
    return [];
  }
}

async function cleanDest(destDir) {
  try {
    await del([path.join(destDir)]);
    await fs.ensureDir(destDir);
  } catch (e) {
    console.error(`Error cleaning destination folder ${destDir}`, e);
  }
}

async function formatFiles(srcDir, destDir, files) {
  try {
    for (const file of files) {
      const lang = await importFile(path.join(srcDir, file));
      const formatted = await formatLang(lang);
      const json = await formatJson(formatted);
      await writeJsonToFile(path.join(destDir, file), json);
    }
  } catch (e) {
    console.error(`Error formatting files:`, e);
  }
}

async function importFile(filepath) {
  try {
    const lang = await require(filepath);
    return lang;
  } catch (e) {
    console.error(`Error importing file ${filepath}`, e);
    return {};
  }
}

async function formatLang(lang) {
  try {
    const keys = Object.keys(lang).sort();
    const formatted = keys.reduce((obj, key) => {
      obj[key] = { defaultMessage: lang[key] };
      return obj;
    }, {});
    return formatted;
  } catch (e) {
    console.error(`Error formatting language object:`, e);
    return {};
  }
}

async function formatJson(obj) {
  try {
    const options = { parser: 'json' };
    const json = await prettier.format(JSON.stringify(obj), options);
    return json;
  } catch (e) {
    console.error(`Error formatting JSON:`, e);
    return '';
  }
}

async function writeJsonToFile(filepath, json) {
  try {
    await fs.writeFile(filepath, json);
  } catch (e) {
    console.error(`Error writing JSON to file ${filepath}`, e);
  }
}

main();