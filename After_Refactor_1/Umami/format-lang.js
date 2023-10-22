const fs = require('fs-extra');
const path = require('path');
const del = require('del');
const prettier = require('prettier');

const srcDir = path.resolve(__dirname, '../lang');
const destDir = path.resolve(__dirname, '../build/messages');

const files = fs.readdirSync(srcDir);

async function formatFiles() {
  // Ensure that the destination directory exists
  await fs.ensureDir(destDir);

  // Delete any existing files in the destination directory
  del.sync([path.join(destDir)]);

  // Format each file in the source directory
  for (const file of files) {
    const langObj = require(`${srcDir}/${file}`);
    const keys = Object.keys(langObj).sort();

    const formattedObj = keys.reduce((obj, key) => {
      obj[key] = { defaultMessage: langObj[key] };
      return obj;
    }, {});

    const formattedJson = prettier.format(JSON.stringify(formattedObj), {
      parser: 'json',
    });

    fs.writeFileSync(path.resolve(destDir, file), formattedJson);
  }
}

async function main() {
  try {
    await formatFiles();
    console.log('Files formatted successfully.');
  } catch (error) {
    console.error('An error occurred while formatting the files:', error);
  }
}

main();

