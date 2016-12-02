#include <queue>
#include <limits>
#include <cmath>
#include <iostream>

class Node {
  public:
    int idx;
    float cost;

    Node(int i, float c) : idx(i),cost(c) {}
};

// the top of the priority queue is the greatest element
bool operator<(const Node &n1, const Node &n2) {
  return n1.cost > n2.cost;
}

bool operator==(const Node &n1, const Node &n2) {
  return n1.idx == n2.idx;
}

float heuristic(int i0, int j0, int i1, int j1) {
  return std::abs(i0 - i1) + std::abs(j0 - j1);
}

std::vector<int> get_neighbors(int idx, int height, int width) {
  int i = idx / width;
  int j = idx % width;

  std::vector<int> neighbors;
  if ((i - 1) >= 0)
    neighbors.push_back((i - 1) * width + j);
  if ((j - 1) >= 0)
    neighbors.push_back(i * width + (j - 1));
  if ((i + 1) < height)
    neighbors.push_back((i + 1) * width + j);
  if ((j + 1) < width)
    neighbors.push_back(i * width + (j + 1));

  return neighbors;
}

extern "C" float astar(
      const float* weights, int height, int width,
      int start, int goal, int* paths) {

  const float INF = std::numeric_limits<float>::infinity();

  Node start_node(start, 0.);
  Node goal_node(goal, 0.);

  float* costs = new float[height * width];
  for (int i = 0; i < height * width; ++i)
    costs[i] = INF;
  costs[start] = 0.;

  std::priority_queue<Node> nodes_to_visit; 
  nodes_to_visit.push(start_node);

  while (!nodes_to_visit.empty()) {
    Node current = nodes_to_visit.top();

    if (current == goal_node) {
      break;
    }

    nodes_to_visit.pop();

    std::vector<int> nbrs = get_neighbors(current.idx, height, width);
    for (std::vector<int>::const_iterator nitr = nbrs.begin(); 
         nitr != nbrs.end(); ++nitr) {
      float new_cost = costs[current.idx] + weights[*nitr];
      if (new_cost < costs[*nitr]) {
        costs[*nitr] = new_cost;
        float priority = new_cost + heuristic((*nitr) / width, 
                                              (*nitr) % width, 
                                              goal / width, 
                                              goal % width);
        nodes_to_visit.push(Node(*nitr, priority));
        paths[*nitr] = current.idx;
      }
    }
  }

  delete[] costs;

  return 0.;
}
