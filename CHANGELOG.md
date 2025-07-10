# Changelog

## [Version 1.1.16] - Bugfix Release - 2025-07
* Fixed feature handling with features with <= 2 characters

## [Version 1.1.15] - Bugfix Release - 2025-06
* Fixed broken plugin icon
* Fix code env package version conflicts
  
## [Version 1.1.14] - Bugfix Release - 2025-06
* Fixed issue with Snowflake model registry input data types.

## [Version 1.1.13] - Bugfix Release - 2025-06
* Fixed snowflake-connector-python version issue.

## [Version 1.1.12] - Bugfix Release - 2025-03
* Fixed LogisticRegression handling of multi-class (due to sklearn rollback).

## [Version 1.1.11] - Bugfix Release - 2025-03
* Changed scikit-learn version to 1.4.2

## [Version 1.1.10] - Bugfix Release - 2025-02
* Added notes throughout the plugin and documentation that a Snowpark-optimized warehouse is required for train.

## [Version 1.1.9] - Bugfix Release - 2025-01
* Fixed package versions for the managed plugin code environment and "py_39_snowpark" code environment, fixing the error that prevented the deployment of models to Snowflake ML Model Registry due to Snowflake removing certain package versions from their internal repository. Upgrading the plugin will prompt rebuilding the managed plugin code environment. You will need to update the "py_39_snowpark" code environment manually accoring to the updated README.

## [Version 1.1.8] - Bugfix Release - 2024-07
* Fixed deployment of models to Snowflake ML Model Registry to control for sklearn backwards incompatibility on the Snowflake side

## [Version 1.1.7] - Bugfix Release - 2024-07

* Fixed error message saying no algorithms selected when only logistic regression selected

## [Version 1.1.6] - Bugfix Release - 2024-07

* Updated Snowpark ML version to 1.5.3
* Fixed Model Registry deletion macro

## [Version 1.1.5] - Bugfix Release - 2024-05

* Updated creation of Snowpark session objects to account for variables in Snowflake connection-level database and schema fields

## [Version 1.1.4] - Initial Release - 2024-05

* Train recipe for visual ML on Snowpark
* Score recipe on Snowpark (using models trained with the Train recipe, and succesfully deployed to Snowflake Model Registry)
* Macro to delete models from Snowflake Model Registry which have been deleted from Dataiku project flows
