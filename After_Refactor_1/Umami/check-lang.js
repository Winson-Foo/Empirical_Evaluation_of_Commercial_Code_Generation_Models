/* eslint-disable no-console */
const fs = require('fs').promises;
const path = require('path');
const chalk = require('chalk');
const messages = require('../lang/en-US.json');
const ignore = require('../lang-ignore.json');

const langDir = path.join(__dirname, '..', 'lang');

async function readJSONFile(dirPath) {
  try {
    const content = await fs.readFile(dirPath, 'utf-8');
    return JSON.parse(content);
  } catch (error) {
    console.error(`Error reading file: ${dirPath}:`, error);
    return {};
  }
}

async function checkLanguageFile(file, keys, filter) {
  const id = file.replace('.json', '');

  if (filter && filter !== id) {
    return;
  }

  console.log(chalk.yellowBright(`\n## ${id}`));
  let count = 0;

  const lang = await readJSONFile(path.join(langDir, file));

  keys.forEach((key) => {
    const orig = messages[key];
    const check = lang[key];
    const ignored = ignore[id] === '*' || ignore[id]?.includes(key);

    if (!ignored && (!check || check === orig)) {
      console.log(chalk.redBright('*'), chalk.greenBright(`${key}:`), orig);
      count++;
    }
  });

  if (count === 0) {
    console.log('**Complete!**');
  }
}

async function main() {
  const files = await fs.readdir(langDir);
  const keys = Object.keys(messages).sort();
  const filter = process.argv?.[2];

  files.some(async (file) => {
    if (file !== 'en-US.json') {
      await checkLanguageFile(file, keys, filter);
    }
  });
}

main().catch((error) => {
  console.error('Error:', error);
});

