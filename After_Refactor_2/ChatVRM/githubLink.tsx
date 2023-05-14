import { buildUrl } from "@/utils/buildUrl";
import styles from "./GitHubLink.module.css";

const GITHUB_URL = "https://github.com/pixiv/ChatVRM";
const GITHUB_TEXT = "Fork me";

export const GitHubLink = () => {
  return (
    <div className={`${styles.container} absolute right-0 z-10 m-24`}>
      <a
        draggable={false}
        href={GITHUB_URL}
        rel="noopener noreferrer"
        target="_blank"
      >
        <div className={`${styles.button} p-8 rounded-16 flex`}>
          <img
            alt={GITHUB_URL}
            height={24}
            width={24}
            src={buildUrl("/github-mark-white.svg")}
          ></img>
          <div className={`${styles.text} mx-4 font-M_PLUS_2 font-bold`}>
            {GITHUB_TEXT}
          </div>
        </div>
      </a>
    </div>
  );
};

And in `GitHubLink.module.css`:

.container {
  right: 0;
  z-index: 10;
  margin: 24px;
}

.button {
  background-color: #1f2328;
  &:hover {
    background-color: #33383e;
  }
  &:active {
    background-color: #565a60;
  }
}

.text {
  color: white;
}

