// TeamMemberRemoveButton.js

import React from 'react';

type TeamMemberRemoveButtonProps = {
  teamId: string;
  userId: string;
  disabled: boolean;
  onSave: () => void;
};

function TeamMemberRemoveButton({ teamId, userId, disabled, onSave }: TeamMemberRemoveButtonProps) {
  const handleButtonClick = () => {
    // TODO: Remove team member
    onSave();
  };

  return (
    <button onClick={handleButtonClick} disabled={disabled}>
      Remove
    </button>
  );
}

export default TeamMemberRemoveButton;