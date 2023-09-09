package org.elasticsearch.xpack.core.ilm.action;

import java.util.Arrays;
import org.elasticsearch.action.support.IndicesOptions;
import org.elasticsearch.common.io.stream.Writeable;
import org.elasticsearch.test.AbstractWireSerializingTestCase;
import org.elasticsearch.xpack.core.ilm.action.RetryAction.Request;

public class RetryActionTests extends AbstractWireSerializingTestCase<Request> {

  @Override
  protected Request createTestInstance() {
    Request request = new Request();
    if (randomBoolean()) {
      request.indices(generateRandomStringArray());
    }
    if (randomBoolean()) {
      request.indicesOptions(generateIndicesOptions());
    }
    return request;
  }

  @Override
  protected Writeable.Reader<Request> instanceReader() {
    return Request::new;
  }

  @Override
  protected Request mutateInstance(Request instance) {
    String[] indices = instance.indices();
    IndicesOptions indicesOptions = instance.indicesOptions();
    int randomCase = between(0, 1);
    switch (randomCase) {
      case 0:
        indices =
            randomValueOtherThanMany(
                i -> Arrays.equals(i, instance.indices()),
                () -> generateRandomStringArray()
            );
        break;
      case 1:
        indicesOptions =
            randomValueOtherThan(
                indicesOptions, () -> generateIndicesOptions()
            );
        break;
      default:
        throw new AssertionError("Illegal randomisation branch");
    }
    Request newRequest = new Request();
    newRequest.indices(indices);
    newRequest.indicesOptions(indicesOptions);
    return newRequest;
  }

  private String[] generateRandomStringArray() {
    return generateRandomStringArray(20, 20, false);
  }

  private IndicesOptions generateIndicesOptions() {
    return IndicesOptions.fromOptions(
        randomBoolean(),
        randomBoolean(),
        randomBoolean(),
        randomBoolean(),
        randomBoolean(),
        randomBoolean(),
        randomBoolean(),
        randomBoolean()
    );
  }
}

