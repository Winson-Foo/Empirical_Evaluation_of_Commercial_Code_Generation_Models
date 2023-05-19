/**
 * This component renders a link with a specified URL and label.
 * @param {string} url - The URL the link should navigate to.
 * @param {string} label - The label to display on the link.
 */
export const Link = ({ url, label }: { url: string; label: string }) => {
  return (
    <a
      className="text-primary hover:text-primary-hover"
      target="_blank"
      rel="noopener noreferrer"
      href={url}
    >
      {label}
    </a>
  );
}; 