// To improve the maintainability of the codebase, we can make the following changes:

// 1. Use arrow functions to improve code readability and make the codebase consistent.
// 2. Use const instead of let for variables that are not reassigned.
// 3. Remove the unnecessary conversion of nodeA and nodeB to strings since the input is already assumed to be strings.
// 4. Move the `dfsTraverse` function inside the `sortAndGetOrderedItems` function to encapsulate it, as it is only used within that function.
// 5. Use a more descriptive and meaningful variable name instead of `finishTimeCount`.
// 6. Organize the code by grouping related functions and variables together.

// Here is the refactored code:

// ```javascript
export function TopologicalSorter() {
  const graph = {};

  this.addOrder = (nodeA, nodeB) => {
    graph[nodeA] = graph[nodeA] || [];
    graph[nodeA].push(nodeB);
  };

  this.sortAndGetOrderedItems = () => {
    const isVisitedNode = Object.create(null);
    const finishingTimeList = [];

    const dfsTraverse = (node, finishTime) => {
      isVisitedNode[node] = true;
      if (graph[node]) {
        graph[node].forEach((nextNode) => {
          if (isVisitedNode[nextNode]) return;
          dfsTraverse(nextNode, finishTime);
        });
      }

      finishingTimeList.push({
        node,
        finishTime: ++finishTime,
      });
    };

    let finishTime = 0;
    for (let node in graph) {
      if (graph.hasOwnProperty(node) && !isVisitedNode[node]) {
        dfsTraverse(node, finishTime);
      }
    }

    finishingTimeList.sort((item1, item2) =>
      item1.finishTime > item2.finishTime ? -1 : 1
    );

    return finishingTimeList.map((value) => value.node);
  };
}

/* TEST */
// const topoSorter = new TopologicalSorter();
// topoSorter.addOrder(5, 2);
// topoSorter.addOrder(5, 0);
// topoSorter.addOrder(4, 0);
// topoSorter.addOrder(4, 1);
// topoSorter.addOrder(2, 3);
// topoSorter.addOrder(3, 1);
// topoSorter.sortAndGetOrderedItems();
// ```

// With these changes, the codebase becomes more maintainable and easier to read and understand.

