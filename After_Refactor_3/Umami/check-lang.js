/* eslint-disable no-console */
const fs = require('fs');
const path = require('path');
const chalk = require('chalk');

function readLanguageFiles() {
  const dir = path.resolve(__dirname, '../lang');
  const files = fs.readdirSync(dir);
  const languageData = {};

  files.forEach(file => {
    if (file !== 'en-US.json') {
      const lang = require(`../lang/${file}`);
      const id = file.replace('.json', '');
      languageData[id] = lang;
    }
  });

  return languageData;
}

function compareMessages(messages, languageData, ignoreList) {
  const keys = Object.keys(messages).sort();
  const diffs = [];

  Object.keys(languageData).forEach(id => {
    const lang = languageData[id];

    keys.forEach(key => {
      const orig = messages[key];
      const check = lang[key];
      const ignored = ignoreList[id] === '*' || ignoreList[id]?.includes(key);

      if (!ignored && (!check || check === orig)) {
        diffs.push({ id, key, orig });
      }
    });
  });

  return diffs;
}

function printDiffs(diffs) {
  diffs.forEach(({ id, key, orig }) => {
    console.log(chalk.yellowBright(`\n## ${id}`));
    console.log(chalk.redBright('*'), chalk.greenBright(`${key}:`), orig);
  });

  if (diffs.length === 0) {
    console.log('**Complete!**');
  }
}

function main() {
  const args = process.argv.slice(2);
  const ignore = require('../lang-ignore.json');
  const messages = require('../lang/en-US.json');
  const languageData = readLanguageFiles();
  const ignoreList = ignore || {};

  try {
    let diffs = compareMessages(messages, languageData, ignoreList);

    if (args.length > 0 && languageData[args[0]]) {
      diffs = diffs.filter(({ id }) => id === args[0]);
    }

    printDiffs(diffs);
  } catch (err) {
    console.error(chalk.redBright('Error:', err.message));
  }
}

main();