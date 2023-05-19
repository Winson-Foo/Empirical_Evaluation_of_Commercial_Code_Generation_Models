To improve the maintainability of this codebase, we can:

1. Extract the GitHub repository URL as a variable, to make it easier to update in case it changes in the future.
2. Extract the styles of the GitHubLink component into a separate CSS file or component, to make it more modular and reusable.
3. Rename the GitHubLink component to something more descriptive, such as "GitHubForkButton".

Here's the refactored code:

import { buildUrl } from "@/utils/buildUrl";

const REPO_URL = "https://github.com/pixiv/ChatVRM";

export const GitHubForkButton = () => {
  return (
    <div className="github-fork-button">
      <a draggable={false} href={REPO_URL} rel="noopener noreferrer" target="_blank">
        <div className="github-fork-button__inner">
          <img
            alt={REPO_URL}
            height={24}
            width={24}
            src={buildUrl("/github-mark-white.svg")}
          />
          <div className="github-fork-button__text">Fork me</div>
        </div>
      </a>
    </div>
  );
};

/* CSS for the GitHubForkButton component */
.github-fork-button {
  position: absolute;
  right: 0;
  z-index: 10;
  margin: 24px;
}

.github-fork-button__inner {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 8px;
  border-radius: 16px;
  background-color: #1f2328;
  cursor: pointer;
}

.github-fork-button__inner:hover {
  background-color: #33383e;
}

.github-fork-button__inner:active {
  background-color: #565a60;
}

.github-fork-button__text {
  margin-left: 4px;
  font-family: "M PLUS 2", sans-serif;
  font-weight: bold;
  color: white;
}

