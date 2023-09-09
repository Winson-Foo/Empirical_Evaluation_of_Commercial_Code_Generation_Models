// To improve the maintainability of the codebase, we can make the following changes:

// 1. Extract the calculation of distances into a separate function for better separation of concerns.
// 2. Extract the calculation of cluster centroids into a separate function for better separation of concerns.
// 3. Use meaningful variable and function names to improve code readability.

// Here's the refactored code:

// ```javascript
import * as mtrx from '../../CONSTANT/javascript_algorithms/Matrix';
import euclideanDistance from '../../CONSTANT/javascript_algorithms/euclideanDistance';

/**
 * Classifies the point in space based on k-Means algorithm.
 *
 * @param {number[][]} data - array of dataSet points, i.e. [[0, 1], [3, 4], [5, 7]]
 * @param {number} k - number of clusters
 * @return {number[]} - the class of the point
 */
export default function KMeans(
  data,
  k = 1,
) {
  if (!data) {
    throw new Error('The data is empty');
  }

  const dataDim = data[0].length;
  const clusterCenters = data.slice(0, k);
  const distances = mtrx.zeros([data.length, k]);
  const classes = Array(data.length).fill(-1);

  let iterate = true;

  function calculateDistances() {
    for (let dataIndex = 0; dataIndex < data.length; dataIndex += 1) {
      for (let clusterIndex = 0; clusterIndex < k; clusterIndex += 1) {
        distances[dataIndex][clusterIndex] = euclideanDistance(
          [clusterCenters[clusterIndex]],
          [data[dataIndex]],
        );
      }
      const closestClusterIdx = distances[dataIndex].indexOf(
        Math.min(...distances[dataIndex]),
      );
      if (classes[dataIndex] !== closestClusterIdx) {
        iterate = true;
      }
      classes[dataIndex] = closestClusterIdx;
    }
  }

  function calculateClusterCentroids() {
    for (let clusterIndex = 0; clusterIndex < k; clusterIndex += 1) {
      clusterCenters[clusterIndex] = Array(dataDim).fill(0);
      let clusterSize = 0;
      for (let dataIndex = 0; dataIndex < data.length; dataIndex += 1) {
        if (classes[dataIndex] === clusterIndex) {
          clusterSize += 1;
          for (let dimensionIndex = 0; dimensionIndex < dataDim; dimensionIndex += 1) {
            clusterCenters[clusterIndex][dimensionIndex] += data[dataIndex][dimensionIndex];
          }
        }
      }
      for (let dimensionIndex = 0; dimensionIndex < dataDim; dimensionIndex += 1) {
        clusterCenters[clusterIndex][dimensionIndex] = parseFloat(Number(
          clusterCenters[clusterIndex][dimensionIndex] / clusterSize,
        ).toFixed(2));
      }
    }
  }

  while (iterate) {
    iterate = false;
    calculateDistances();
    calculateClusterCentroids();
  }

  return classes;
}
// ```

// These changes should improve the maintainability of the codebase by separating the logic into smaller and more focused functions. Additionally, using meaningful variable and function names can make the code easier to understand and modify in the future.

