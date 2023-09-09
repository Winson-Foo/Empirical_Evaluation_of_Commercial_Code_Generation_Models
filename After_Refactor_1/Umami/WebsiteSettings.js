import { useEffect, useState } from 'react';
import { Breadcrumbs, Item, Tabs, useToast, Button, Text, Icon, Icons } from 'react-basics';
import { useRouter } from 'next/router';
import Link from 'next/link';
import Page from 'components/layout/Page';
import PageHeader from 'components/layout/PageHeader';
import WebsiteEditForm from 'components/pages/settings/websites/WebsiteEditForm';
import WebsiteData from 'components/pages/settings/websites/WebsiteData';
import TrackingCode from 'components/pages/settings/websites/TrackingCode';
import ShareUrl from 'components/pages/settings/websites/ShareUrl';
import useApi from 'hooks/useApi';
import useMessages from 'hooks/useMessages';
import useConfig from 'hooks/useConfig';

const TABS = [
  {
    key: 'details',
    label: 'Details',
    component: WebsiteEditForm,
  },
  {
    key: 'tracking',
    label: 'Tracking Code',
    component: TrackingCode,
  },
  {
    key: 'share',
    label: 'Share URL',
    component: ShareUrl,
  },
  {
    key: 'data',
    label: 'Data',
    component: WebsiteData,
  },
];

function WebsiteSettings({ websiteId }) {
  const router = useRouter();
  const { formatMessage, labels, messages } = useMessages();
  const { openExternal } = useConfig();
  const { get, useQuery } = useApi();
  const { toast, showToast } = useToast();
  const [values, setValues] = useState(null);
  const [selectedTab, setSelectedTab] = useState(TABS[0].key);

  const showSuccessToast = () => {
    showToast({ message: formatMessage(messages.saved), variant: 'success' });
  };

  const handleSave = (data) => {
    showSuccessToast();
    setValues((prevState) => ({ ...prevState, ...data }));
  };

  const handleReset = async (value) => {
    if (value === 'delete') {
      await router.push('/settings/websites');
    } else if (value === 'reset') {
      showSuccessToast();
    }
  };

  const { data, isLoading } = useQuery(['website', websiteId], () => get(`/websites/${websiteId}`), {
    enabled: !!websiteId,
    cacheTime: 0,
  });

  useEffect(() => {
    if (data) {
      setValues(data);
    }
  }, [data]);

  const renderTabContent = (key) => {
    const { component: TabComponent } = TABS.find((tab) => tab.key === key);

    return <TabComponent websiteId={websiteId} data={values} onSave={handleSave} />;
  };

  return (
    <Page loading={isLoading || !values}>
      {toast}
      <PageHeader
        title={
          <Breadcrumbs>
            <Item>
              <Link href="/settings/websites">{formatMessage(labels.websites)}</Link>
            </Item>
            <Item>{values?.name}</Item>
          </Breadcrumbs>
        }
      >
        <Link href={`/websites/${websiteId}`} target={openExternal ? '_blank' : null}>
          <Button variant="primary">
            <Icon>
              <Icons.External />
            </Icon>
            <Text>{formatMessage(labels.view)}</Text>
          </Button>
        </Link>
      </PageHeader>
      <Tabs selectedKey={selectedTab} onSelect={setSelectedTab} style={{ marginBottom: 30 }}>
        {TABS.map(({ key, label }) => (
          <Item key={key}>{formatMessage(label)}</Item>
        ))}
      </Tabs>
      {renderTabContent(selectedTab)}
    </Page>
  );
}

export default WebsiteSettings;