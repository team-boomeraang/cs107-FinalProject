## Team Boomdiff - Final Milestone

Broadly, there are four areas still left to implement for the final milestone, each of which will require updated testing;

- Optimization library implementation
- Updated documentation/packaging
- Additional forward-mode requirements - functions with multiple vector/scalar inputs
- Deliverables
  - Broader impact statement
  - Video

The breakdown for grading for these sections is as follows:

| Points | Task                                 |
| ------ | ------------------------------------ |
| 20     | Complete forward mode implementation |
| 15     | Documentation & Test Suite           |
| 4      | Broader Impact Statement             |
| 20     | Video presentation                   |
| 30     | New feature (optimization library)   |
| 15     | Code quality                         |

In terms of tasks remaining, we have the following:

##### Forward mode implementation

| Task                                                         | Team member | Status   |
| ------------------------------------------------------------ | ----------- | -------- |
| Update existing functions for multiple inputs, where applicable | Minhuan     | Complete |
| `__eq__`, `__neq__`, other comparison operators              | Kevin       | Complete |
| inverse trig functions (arcsine, arctan, arccos)             | Oksana      | Complete |
| hyperbolic functions (sinh, tanh, cosh)                      | Timothy     | Complete |
| logistic function                                            | Kevin       | Complete |
| logarithm - expand to any base with default of e             | Timothy     | Complete |
| square root                                                  | Timothy     | Complete |
| Update base class to support default seed vector             | Kevin       | Complete |

##### Documentation, software organization, deliverables

| Task                                                         | Team member | Status      |
| ------------------------------------------------------------ | ----------- | ----------- |
| Discuss reorganization suggested by David                    | All         | Complete    |
| (TBD) Reorganize software package for optimization           | Kevin       | Complete    |
| Add extension documentation to reflect implementation details |             |             |
| Revise background & how-to-use sections                      |             |             |
| Broader impact statement                                     | Timothy     |             |
| Plan video & record                                          | All         |             |
| Make package pip-installable                                 | Kevin       | In progress |

##### Optimization library

| Task                                                         | Team member | Status      |
| ------------------------------------------------------------ | ----------- | ----------- |
| Base objective function class                                | Minhuan     | In progress |
| 'set_params' method                                          | Minhuan     |             |
| 'optimize' method                                            | Oksana      |             |
| Implement gradient descent optimization                      | Oksana      |             |
| Implement random coordinate descent (or other opt. algorithm) | TBD         |             |
| Implement BFGS optimization                                  | Minhuan     |             |

