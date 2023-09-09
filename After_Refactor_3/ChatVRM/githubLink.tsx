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