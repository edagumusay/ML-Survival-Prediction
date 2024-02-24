# Titanic Survival Prediction - Classification

Predictions will be made on survival outcomes based on the Titanic dataset.

The target variable, whether a passenger survived or not, is a binary outcome (0 or 1), making it suitable for classification modeling. Therefore, classification algorithms are appropriate for this task.

<h3>Data Dictionary</h3>
<table>
  <tr>
    <th>Variable</th>
    <th>Definition</th>
    <th>Key</th>
  </tr>
  <tr>
    <td>Survival</td>
    <td>Survival</td>
    <td>0 = No, 1 = Yes</td>
  </tr>
  <tr>
    <td>Pclass</td>
    <td>Ticket class</td>
    <td>1 = 1st, 2 = 2nd, 3 = 3rd</td>
  </tr>
  <tr>
    <td>Sex</td>
    <td>Gender of the passenger</td>
    <td></td>
  </tr>
  <tr>
    <td>Age</td>
    <td>Age in years</td>
    <td></td>
  </tr>
  <tr>
    <td>SibSp</td>
    <td>Number of siblings / spouses aboard the Titanic</td>
    <td></td>
  </tr>
  <tr>
    <td>Parch</td>
    <td>Number of parents / children aboard the Titanic</td>
    <td></td>
  </tr>
  <tr>
    <td>Ticket</td>
    <td>Ticket number</td>
    <td></td>
  </tr>
  <tr>
    <td>Fare</td>
    <td>Passenger fare</td>
    <td></td>
  </tr>
  <tr>
    <td>Cabin</td>
    <td>Cabin number</td>
    <td></td>
  </tr>
  <tr>
    <td>Embarked</td>
    <td>Port of Embarkation</td>
    <td>C = Cherbourg, Q = Queenstown, S = Southampton</td>
  </tr>
</table>


### Variable Notes
* Pclass: A proxy for socio-economic status (SES)
  * 1st = Upper
  * 2nd = Middle
  * 3rd = Lower
* Age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
* SibSp: The dataset defines family relations in this way...
  * Sibling = brother, sister, stepbrother, stepsister
  * Spouse = husband, wife (mistresses and fiancés were ignored)
* Parch: The dataset defines family relations in this way...
  * Parent = mother, father
  * Child = daughter, son, stepdaughter, stepson
      * Some children travelled only with a nanny, therefore parch=0 for them.
