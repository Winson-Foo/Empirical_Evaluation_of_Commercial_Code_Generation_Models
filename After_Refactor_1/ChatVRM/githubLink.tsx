import { buildUrl } from "@/utils/buildUrl";

/**
 * A component that displays a link to the GitHub repository of the project.
 */
export const GitHubLink = () => {
  return (
    <div className="absolute right-0 z-10 m-24">
      <a
        // Disable dragging to prevent accidental drag-and-drop actions.
        draggable={false} 
        href="https://github.com/pixiv/ChatVRM"
        // Open the link in a new tab.
        target="_blank" 
        // Use noopener and noreferrer for improved security.
        rel="noopener noreferrer" 
      >
        <div 
          // Use a meaningful class name for the container element.
          className="github-link-container"
        >
          <img
            // Add an alt attribute for accessibility.
            alt="GitHub repository link"
            height={24}
            width={24}
            // Use a helper function to generate the image URL.
            src={buildUrl("/github-mark-white.svg")}
          ></img>
          <div 
            // Use a meaningful class name for the text element.
            className="github-link-text"
          >
            Fork me
          </div>
        </div>
      </a>
    </div>
  );
};

