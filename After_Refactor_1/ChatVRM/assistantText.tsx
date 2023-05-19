// AssistantText.tsx
import React from "react";
import AssistantHeader from "./AssistantHeader";
import AssistantMessage from "./AssistantMessage";

interface Props {
  message: string;
}

const AssistantText: React.FC<Props> = ({ message }) => {
  return (
    <div className="absolute bottom-0 left-0 mb-104 w-full">
      <div className="mx-auto max-w-4xl w-full p-16">
        <div className="bg-white rounded-8">
          <AssistantHeader />
          <AssistantMessage message={message} />
        </div>
      </div>
    </div>
  );
};

export default AssistantText;

// AssistantHeader.tsx
import React from "react";

const AssistantHeader: React.FC = () => {
  return (
    <div className="px-24 py-8 bg-secondary rounded-t-8 text-white font-Montserrat font-bold tracking-wider">
      CHARACTER
    </div>
  );
};

export default AssistantHeader;

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