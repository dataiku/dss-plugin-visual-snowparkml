const app = angular.module('trainSnowparkModel.recipe', []);

app.controller('retrainRecipeController', function ($scope, utils) {

    $scope.selectedInputColumns = {};
    $scope.inputColumnTypes = {};
    $scope.showOptions = {};
    $scope.validationMessage = '';
    
    const updateCommonScopeData = function (data) {
        $scope.styleSheetUrl = utils.getStylesheetUrl(data.pluginId);
    }
    
    const updateScopeData = function (data) {
              
        updateCommonScopeData(data)
        $scope.config.inputDatasetColumns = data.input_dataset_columns;
        $scope.inputDatasetColNames = data.input_dataset_columns.map(col => col.name);
                
        $scope.categoricalOptions = ["Dummy encoding", "Ordinal encoding"];
        $scope.numericOptions = ["Standard rescaling","Min-max rescaling","No rescaling"];
        $scope.categoricalOptions2 = ["Most frequent value", "Constant"];
        $scope.numericOptions2 = ["Average", "Median", "Constant"];
        
        $scope.updateSelectedOptions = function (columnName, optionNum) {
            const selectedOptionKey = "selectedOption" + optionNum;
            const selectedOption = $scope[selectedOptionKey][columnName];
            const isFeatureSelected = selectedOption !== "";
            $scope.showOptions[columnName] = isFeatureSelected;
        };

        $scope.toggleFeatureSelection = function (columnName) {
            const isFeatureSelected = $scope.config.selectedInputColumns[columnName];
            if (isFeatureSelected) {
                // Feature is being turned ON - set default values only if not already set
                const column = $scope.config.inputDatasetColumns.find(c => c.name === columnName);
                if (column) {
                    const isNumeric = column.type.indexOf('int') !== -1 ||
                                      column.type.indexOf('float') !== -1 ||
                                      column.type.indexOf('double') !== -1;

                    // Initialize objects if needed
                    if (!$scope.config.selectedOption1) {
                        $scope.config.selectedOption1 = {};
                    }
                    if (!$scope.config.selectedOption2) {
                        $scope.config.selectedOption2 = {};
                    }

                    // Only set default if no value exists for this column
                    if (!$scope.config.selectedOption1[columnName]) {
                        $scope.config.selectedOption1[columnName] = isNumeric ?
                            $scope.numericOptions[0] : $scope.categoricalOptions[0];
                    }

                    if (!$scope.config.selectedOption2[columnName]) {
                        $scope.config.selectedOption2[columnName] = isNumeric ?
                            $scope.numericOptions2[0] : $scope.categoricalOptions2[0];
                    }
                }
                $scope.showOptions[columnName] = true;
            } else {
                // Feature is being turned OFF - keep the values, just hide options
                $scope.showOptions[columnName] = false;
            }
        };

    };    
    
    $scope.metrics = [];

    $scope.updateMetrics = function() {
        $scope.metrics = [];

        if ($scope.config.prediction_type === 'two-class classification') {
            $scope.metrics = ['ROC AUC', 'F1 Score', 'Accuracy', 'Precision', 'Recall'];
            // Set default metric if not already set or if switching from another prediction type
            if (!$scope.config.model_metric || !$scope.metrics.includes($scope.config.model_metric)) {
                $scope.config.model_metric = 'ROC AUC';
            }
            $scope.config.lasso_regression = false;
            $scope.config.random_forest_regression = false;
            $scope.config.xgb_regression = false;
            $scope.config.lgbm_regression = false;
            $scope.config.gb_regression = false;
            $scope.config.decision_tree_regression = false;
        } else if ($scope.config.prediction_type === 'multi-class classification') {
            $scope.metrics = ['ROC AUC', 'F1 Score', 'Accuracy', 'Precision', 'Recall'];
            // Set default metric if not already set or if switching from another prediction type
            if (!$scope.config.model_metric || !$scope.metrics.includes($scope.config.model_metric)) {
                $scope.config.model_metric = 'ROC AUC';
            }
            $scope.config.lasso_regression = false;
            $scope.config.random_forest_regression = false;
            $scope.config.xgb_regression = false;
            $scope.config.lgbm_regression = false;
            $scope.config.gb_regression = false;
            $scope.config.decision_tree_regression = false;
        } else if ($scope.config.prediction_type === 'regression') {
            $scope.metrics = ['R2', 'MAE', 'MSE'];
            // Set default metric if not already set or if switching from another prediction type
            if (!$scope.config.model_metric || !$scope.metrics.includes($scope.config.model_metric)) {
                $scope.config.model_metric = 'R2';
            }
            $scope.config.logistic_regression = false;
            $scope.config.random_forest_classification = false;
            $scope.config.xgb_classification = false;
            $scope.config.lgbm_classification = false;
            $scope.config.gb_classification = false;
            $scope.config.decision_tree_classification = false;
        }
    };
    
    $scope.updateMetrics();

    $scope.canNavigateAway = function() {
        return $scope.config.prediction_type &&
               $scope.config.col_label &&
               $scope.config.model_name &&
               $scope.config.model_name.trim() !== '';
    };

    $scope.changeTab = function(newTab) {
        if (newTab === 'target') {
            $scope.activeTab = newTab;
            $scope.validationMessage = '';
            return;
        }

        if (!$scope.canNavigateAway()) {
            $scope.validationMessage = 'Please complete all required fields in the Target tab: Prediction type, Target column, and Model name.';
            return;
        }

        $scope.activeTab = newTab;
        $scope.validationMessage = '';
    };

    const init = function () {
        $scope.finishedLoading = false;
        $scope.updateMetrics();
        utils.retrieveInfoBackend($scope, "get-info-retrain", updateScopeData);

        // Initialize feature option objects if not present
        if (!$scope.config.selectedOption1) {
            $scope.config.selectedOption1 = {};
        }
        if (!$scope.config.selectedOption2) {
            $scope.config.selectedOption2 = {};
        }
        if (!$scope.config.selectedInputColumns) {
            $scope.config.selectedInputColumns = {};
        }

        utils.initVariable($scope, 'col_label', '');
        utils.initVariable($scope, 'time_ordering_variable', '');
        utils.initVariable($scope, 'train_ratio', 0.8);
        utils.initVariable($scope, 'random_seed', 42);
        
        utils.initVariable($scope, 'logistic_regression_c_min', 0.01);
        utils.initVariable($scope, 'logistic_regression_c_max', 100);
        
        utils.initVariable($scope, 'random_forest_classification_n_estimators_min', 80);
        utils.initVariable($scope, 'random_forest_classification_n_estimators_max', 200);
        utils.initVariable($scope, 'random_forest_classification_max_depth_min', 6);
        utils.initVariable($scope, 'random_forest_classification_max_depth_max', 20);
        utils.initVariable($scope, 'random_forest_classification_min_samples_leaf_min', 1);
        utils.initVariable($scope, 'random_forest_classification_min_samples_leaf_max', 20);
        
        utils.initVariable($scope, 'xgb_classification_n_estimators_min', 200);
        utils.initVariable($scope, 'xgb_classification_n_estimators_max', 400);
        utils.initVariable($scope, 'xgb_classification_max_depth_min', 3);
        utils.initVariable($scope, 'xgb_classification_max_depth_max', 10);
        utils.initVariable($scope, 'xgb_classification_min_child_weight_min', 1);
        utils.initVariable($scope, 'xgb_classification_min_child_weight_max', 5);
        utils.initVariable($scope, 'xgb_classification_learning_rate_min', 0.01);
        utils.initVariable($scope, 'xgb_classification_learning_rate_max', 0.2);
        
        utils.initVariable($scope, 'lgbm_classification_n_estimators_min', 200);
        utils.initVariable($scope, 'lgbm_classification_n_estimators_max', 400);
        utils.initVariable($scope, 'lgbm_classification_max_depth_min', 3);
        utils.initVariable($scope, 'lgbm_classification_max_depth_max', 10);
        utils.initVariable($scope, 'lgbm_classification_min_child_weight_min', 1);
        utils.initVariable($scope, 'lgbm_classification_min_child_weight_max', 5);
        utils.initVariable($scope, 'lgbm_classification_learning_rate_min', 0.01);
        utils.initVariable($scope, 'lgbm_classification_learning_rate_max', 0.2);
        
        utils.initVariable($scope, 'gb_classification_n_estimators_min', 200);
        utils.initVariable($scope, 'gb_classification_n_estimators_max', 400);
        utils.initVariable($scope, 'gb_classification_max_depth_min', 3);
        utils.initVariable($scope, 'gb_classification_max_depth_max', 10);
        utils.initVariable($scope, 'gb_classification_min_samples_leaf_min', 1);
        utils.initVariable($scope, 'gb_classification_min_samples_leaf_max', 20);
        utils.initVariable($scope, 'gb_classification_learning_rate_min', 0.01);
        utils.initVariable($scope, 'gb_classification_learning_rate_max', 0.2);
        
        utils.initVariable($scope, 'decision_tree_classification_max_depth_min', 3);
        utils.initVariable($scope, 'decision_tree_classification_max_depth_max', 10);
        utils.initVariable($scope, 'decision_tree_classification_min_samples_leaf_min', 1);
        utils.initVariable($scope, 'decision_tree_classification_min_samples_leaf_max', 20);
        
        utils.initVariable($scope, 'lasso_regression_alpha_min', 0.1);
        utils.initVariable($scope, 'lasso_regression_alpha_max', 10);
        
        utils.initVariable($scope, 'random_forest_regression_n_estimators_min', 80);
        utils.initVariable($scope, 'random_forest_regression_n_estimators_max', 200);
        utils.initVariable($scope, 'random_forest_regression_max_depth_min', 6);
        utils.initVariable($scope, 'random_forest_regression_max_depth_max', 20);
        utils.initVariable($scope, 'random_forest_regression_min_samples_leaf_min', 1);
        utils.initVariable($scope, 'random_forest_regression_min_samples_leaf_max', 20);
        
        utils.initVariable($scope, 'xgb_regression_n_estimators_min', 200);
        utils.initVariable($scope, 'xgb_regression_n_estimators_max', 400);
        utils.initVariable($scope, 'xgb_regression_max_depth_min', 3);
        utils.initVariable($scope, 'xgb_regression_max_depth_max', 10);
        utils.initVariable($scope, 'xgb_regression_min_child_weight_min', 1);
        utils.initVariable($scope, 'xgb_regression_min_child_weight_max', 5);
        utils.initVariable($scope, 'xgb_regression_learning_rate_min', 0.01);
        utils.initVariable($scope, 'xgb_regression_learning_rate_max', 0.2);
        
        utils.initVariable($scope, 'lgbm_regression_n_estimators_min', 200);
        utils.initVariable($scope, 'lgbm_regression_n_estimators_max', 400);
        utils.initVariable($scope, 'lgbm_regression_max_depth_min', 3);
        utils.initVariable($scope, 'lgbm_regression_max_depth_max', 10);
        utils.initVariable($scope, 'lgbm_regression_min_child_weight_min', 1);
        utils.initVariable($scope, 'lgbm_regression_min_child_weight_max', 5);
        utils.initVariable($scope, 'lgbm_regression_learning_rate_min', 0.01);
        utils.initVariable($scope, 'lgbm_regression_learning_rate_max', 0.2);
        
        utils.initVariable($scope, 'gb_regression_n_estimators_min', 200);
        utils.initVariable($scope, 'gb_regression_n_estimators_max', 400);
        utils.initVariable($scope, 'gb_regression_max_depth_min', 3);
        utils.initVariable($scope, 'gb_regression_max_depth_max', 10);
        utils.initVariable($scope, 'gb_regression_min_samples_leaf_min', 1);
        utils.initVariable($scope, 'gb_regression_min_samples_leaf_max', 20);
        utils.initVariable($scope, 'gb_regression_learning_rate_min', 0.01);
        utils.initVariable($scope, 'gb_regression_learning_rate_max', 0.2);
        
        utils.initVariable($scope, 'decision_tree_regression_max_depth_min', 3);
        utils.initVariable($scope, 'decision_tree_regression_max_depth_max', 10);
        utils.initVariable($scope, 'decision_tree_regression_min_samples_leaf_min', 1);
        utils.initVariable($scope, 'decision_tree_regression_min_samples_leaf_max', 20);
        
        utils.initVariable($scope, 'n_iter', 4);
        utils.initVariable($scope, 'enable_class_weights', true);
        utils.initVariable($scope, 'deploy_to_snowflake_model_registry', true);

    };

    init();
});

app.service("utils", function () {
    
    this.retrieveInfoBackend = function ($scope, method, updateScopeData) {
      $scope.callPythonDo({method}).then(function (data) {
        updateScopeData(data);
        $scope.finishedLoading = true;
      }, function (data) {
        $scope.finishedLoading = true;
      });
    };
    
    this.initVariable = function ($scope, varName, initValue) {
        const isConfigDefined = angular.isDefined($scope.config);
        if (isConfigDefined) {
            const isVarDefined = angular.isDefined($scope.config[varName]);
            if (isVarDefined) {
                // Retain the existing value if it is defined
                initValue = $scope.config[varName];
            }
        } else {
            $scope.config = {};
        }
        $scope.config[varName] = initValue;
    };
    
    this.getStylesheetUrl = function (pluginId) {
        return `/plugins/${pluginId}/resource/stylesheets/visual-snowparkml-stylesheet.css`;
    };
});