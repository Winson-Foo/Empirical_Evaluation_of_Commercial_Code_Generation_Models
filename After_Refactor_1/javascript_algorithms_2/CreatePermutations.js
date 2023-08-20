// To improve the maintainability of the codebase, we can make the following changes:

// 1. Use meaningful variable names: Instead of using single-letter variable names like `arr`, `strLen`, `perms`, etc., use descriptive names that convey the purpose of the variable.

// 2. Split the logic into smaller functions: Splitting the logic into smaller functions will make the code easier to understand and maintain. We can create a helper function to generate permutations for a given set of characters.

// 3. Remove unnecessary comments: Comments that simply restate the obvious or describe what a particular line of code does should be removed, as they add clutter to the code.

// Here's the refactored code with the above improvements:

const generatePermutations = (arr) => {
  const len = arr.length;
  const permutations = [];
  
  if (len === 0) {
    return [arr];
  }
  
  for (let i = 0; i < len; i++) {
    const rest = Object.create(arr);
    const picked = rest.splice(i, 1);
    const restPerms = generatePermutations(rest.join(''));
    
    for (let j = 0, jLen = restPerms.length; j < jLen; j++) {
      const next = picked.concat(restPerms[j]);
      permutations.push(next.join(''));
    }
  }
  
  return permutations;
}

const createPermutations = (str) => {
  const chars = str.split('');
  return generatePermutations(chars);
}

export { createPermutations }

