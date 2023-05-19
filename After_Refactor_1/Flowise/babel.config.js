const typescriptPreset = '@babel/preset-typescript';
const envPreset = [
  '@babel/preset-env',
  {
    targets: {
      node: 'current',
    },
  },
];

module.exports = {
  presets: [typescriptPreset, envPreset],
};