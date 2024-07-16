# Changelog

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
