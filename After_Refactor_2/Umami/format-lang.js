const fs = require('fs-extra');
const path = require('path');
const del = require('del');
const prettier = require('prettier');

const langDir = path.resolve(__dirname, '../lang');
const buildDir = path.resolve(__dirname, '../build/messages');

async function formatLangFiles() {
  try {
    await deleteBuildDir();
    await createBuildDir();
    const langFiles = getLangFiles();
    langFiles.forEach(langFile => {
      const lang = require(path.join(langDir, langFile));
      const formatted = formatMessages(lang);
      const json = prettier.format(JSON.stringify(formatted), { parser: 'json' });
      writeFormattedFile(json, langFile);
    });
    console.log('All files formatted successfully!');
  } catch (error) {
    console.error('Error while formatting files: ', error);
  }
}

async function deleteBuildDir() {
  console.log('Deleting old build directory...');
  await del([buildDir]);
}

async function createBuildDir() {
  console.log('Creating new build directory...');
  await fs.ensureDir(buildDir);
}

function getLangFiles() {
  console.log('Getting language files...');
  return fs.readdirSync(langDir);
}

function formatMessages(lang) {
  console.log(`Formatting messages for ${lang.code}...`);
  const keys = Object.keys(lang).sort();
  return keys.reduce((obj, key) => {
    obj[key] = { defaultMessage: lang[key] };
    return obj;
  }, {});
}

function writeFormattedFile(json, langFile) {
  console.log(`Writing formatted file for ${langFile}...`);
  fs.writeFileSync(path.join(buildDir, langFile), json);
}

formatLangFiles();

