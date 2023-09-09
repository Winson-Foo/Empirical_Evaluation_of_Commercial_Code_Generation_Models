import crypto from 'crypto';
import { v4, v5 } from 'uuid';
import { startOfMonth } from 'date-fns';
import { hash as basicsHash } from 'next-basics';

const appSecret = process.env.APP_SECRET || process.env.DATABASE_URL;

/**
 * Returns a hashed version of the app secret.
 *
 * @returns {string} The hashed app secret.
 */
export const getAppSecretHash = () => {
  return basicsHash(appSecret);
}

/**
 * Returns a rotating salt based on the start of the current month
 * and the hashed app secret.
 *
 * @returns {string} The rotated salt.
 */
export const getRotatingSalt = () => {
  const date = startOfMonth(new Date()).toUTCString();
  const rotatingSalt = basicsHash(date);
  const appSecretHash = getAppSecretHash();

  return basicsHash(appSecretHash, rotatingSalt);
}

/**
 * Generates a UUID v4 or v5 based on the provided arguments.
 *
 * @param {...string} args - The arguments to be hashed to generate the UUID.
 * @returns {string} The generated UUID.
 */
export const generateUUID = (...args) => {
  const salt = getRotatingSalt();
  const hashedArgs = basicsHash(...args, salt);

  if (!args.length) {
    return v4();
  } else {
    return v5(hashedArgs, v5.DNS);
  }
}

/**
 * Generates an MD5 hash for the provided arguments.
 *
 * @param {...string} args - The arguments to be hashed.
 * @returns {string} The MD5 hash.
 */
export const generateMD5 = (...args) => {
  const message = args.join('');
  const md5Hash = crypto.createHash('md5').update(message).digest('hex');

  return md5Hash;
}

