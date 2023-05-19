// LoginPage.jsx

import LoginLayout from 'components/pages/login/LoginLayout';
import LoginForm from 'components/pages/login/LoginForm';

const LoginPage = ({ disabled }) => {
  if (disabled) {
    return null;
  }

  return (
    <LoginLayout title="Login">
      <LoginForm />
    </LoginLayout>
  );
}

export default LoginPage;

// getServerSideProps.js

export async function getServerSideProps() {
  try {
    const disabled = !!process.env.DISABLE_LOGIN;
    return { props: { disabled } };
  } catch (error) {
    console.error('Error fetching server side props', error);
    return { props: { disabled: false } };
  }
}