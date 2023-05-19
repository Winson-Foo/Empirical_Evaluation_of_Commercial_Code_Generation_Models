import crypto from 'crypto';
import { v4, v5 } from 'uuid';
import { startOfMonth } from 'date-fns';
import { hash } from 'next-basics';

/**
 * Generates a secret key using the APP_SECRET or DATABASE_URL environment variable
 * @returns {string} A hashed secret key
 */
export function generateSecretKey() {
  return hash(process.env.APP_SECRET || process.env.DATABASE_URL);
}

/**
 * Generates a salt for use with passwords
 * @returns {string} A hashed salt value
 */
export function generateSalt() {
  const rotatingSalt = hash(startOfMonth(new Date()).toUTCString());
  return hash(generateSecretKey(), rotatingSalt);
}

/**
 * Generates a UUID based on the input arguments and salt
 * @param {...string} args - Input arguments to be hashed
 * @returns {string} A hashed UUID
 */
export function generateUUID(...args) {
  if (!args.length) return v4();
  const salt = generateSalt();
  return v5(hash(...args, salt), v5.DNS);
}

/**
 * Generates an MD5 hash of the input arguments
 * @param {...string} args - Input arguments to be hashed
 * @returns {string} An MD5 hash
 */
export function generateMD5Hash(...args) {
  return crypto.createHash('md5').update(args.join('')).digest('hex');
}

