// AssistantMessage.tsx
import React from "react";

interface Props {
  message: string;
}

const AssistantMessage: React.FC<Props> = ({ message }) => {
  const filteredMessage = message.replace(/\[([a-zA-Z]*?)\]/g, "");

  return (
    <div className="px-24 py-16">
      <div className="line-clamp-4 text-secondary typography-16 font-M_PLUS_2 font-bold">
        {filteredMessage}
      </div>
    </div>
  );
};

export default AssistantMessage; 