const envOptions = {
    targets: {
        node: 'current'
    }
};

const typescriptPreset = '@babel/preset-typescript';
const envPreset = ['@babel/preset-env', envOptions];

module.exports = {
    presets: [
        typescriptPreset,
        envPreset
    ]
};