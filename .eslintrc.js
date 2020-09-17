module.exports = {
  env: {
    browser: true,
    es2021: true,
    node: false,
  },
  extends: ["prettier", "eslint:recommended"],
  parserOptions: {
    ecmaFeatures: {
      jsx: false,
    },
    ecmaVersion: 12,
    sourceType: "module",
  },
  plugins: ["prettier"],
  rules: {
    "prettier/prettier": [
      "error",
      {
        semi: true,
        singleQuote: true,
        trailingComma: "es5",
      },
    ],
  },
  settings: {
    react: {
      version: "detetect",
    },
  },
};
