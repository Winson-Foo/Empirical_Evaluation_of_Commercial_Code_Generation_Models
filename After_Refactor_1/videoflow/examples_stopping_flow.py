def create_flow():
    producer = IntProducer(0, 40, 0.1)
    identity = IdentityProcessor(nb_tasks = 5)(producer)
    identity1 = IdentityProcessor(nb_tasks = 5)(identity)
    joined = JoinerProcessor(nb_tasks = 5)(identity, identity1)
    printer = CommandlineConsumer()(joined)
    flow = Flow([producer], [printer])
    return flow

if __name__ == '__main__':
    flow = create_flow()
    flow.run()
    time.sleep(2)
    flow.stop()
