/* eslint-disable no-console */
const fs = require('fs');
const path = require('path');
const chalk = require('chalk');

const LANG_DIR = path.resolve(__dirname, '../lang');
const messages = require(`${LANG_DIR}/en-US.json`);
const ignore = require(`${LANG_DIR}-ignore.json`);

// Get all language file names from the directory
const getAllLangFiles = () => {
  try {
    return fs.readdirSync(LANG_DIR);
  } catch (error) {
    console.error(chalk.redBright('Unable to read language directory:'), error);
    process.exit(1);
  }
};

// Compare the English messages with the messages in a specific language file
const compareMessages = (langFilePath) => {
  const lang = require(langFilePath);
  const langId = path.basename(langFilePath, '.json');

  console.log(chalk.yellowBright(`\n## ${langId}`));
  let count = 0;

  Object.keys(messages).forEach((key) => {
    const origMsg = messages[key];
    const langMsg = lang[key];
    const isIgnored = ignore[langId] === '*' || ignore[langId]?.includes(key);

    if (!isIgnored && (!langMsg || langMsg === origMsg)) {
      console.log(chalk.redBright('*'), chalk.greenBright(`${key}:`), origMsg);
      count++;
    }
  });

  if (count === 0) {
    console.log('**Complete!**');
  }
};

// Main function that runs the script
const run = () => {
  const filter = process.argv?.[2];
  const langFiles = getAllLangFiles();

  langFiles
    .filter((fileName) => fileName !== 'en-US.json' && (!filter || filter === path.basename(fileName, '.json')))
    .forEach((fileName) => {
      const langFilePath = path.join(LANG_DIR, fileName);
      compareMessages(langFilePath);
    });
};

// Run the script
run();