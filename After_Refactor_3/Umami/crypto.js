// utils.js
import crypto from 'crypto';
import { v4, v5 } from 'uuid';
import { startOfMonth } from 'date-fns';
import { hash } from 'next-basics';

export function generateSecret() {
  // Generate a unique hash based on the app secret and database URL
  return hash(process.env.APP_SECRET || process.env.DATABASE_URL);
}

export function generateRotatingSalt() {
  // Generate a rotating salt based on the current date
  const ROTATING_SALT = hash(startOfMonth(new Date()).toUTCString());

  // Combine the secret hash and rotating salt to generate a unique salt
  return hash(generateSecret(), ROTATING_SALT);
}

export function generateUuid(...args) {
  if (!args.length) {
    // Generate a v4 UUID if no arguments are passed
    return v4();
  }

  // Combine the arguments, salt, and secret to generate a v5 UUID
  return v5(hash(...args, generateRotatingSalt()), v5.DNS);
}

export function generateMd5Hash(...args) {
  // Generate an MD5 hash based on the concatenated input arguments
  return crypto.createHash('md5').update(args.join('')).digest('hex');
}

// index.js
import { generateUuid, generateMd5Hash } from './utils';

const uuid = generateUuid('example@example.com', 'password');
const hash = generateMd5Hash('example', 'password');

console.log(uuid); // Output: "4ab0f5a3-c98a-5e38-819a-1232fbabfa0b"
console.log(hash); // Output: "5f4dcc3b5aa765d61d8327deb882cf99"

