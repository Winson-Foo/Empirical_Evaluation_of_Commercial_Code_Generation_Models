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