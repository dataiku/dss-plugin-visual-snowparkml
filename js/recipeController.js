const app = angular.module('trainSnowparkModel.recipe', []);

app.controller('retrainRecipeController', function ($scope, utils) {
    
    $scope.selectedInputColumns = {};
    $scope.inputColumnTypes = {};
    $scope.showOptions = {};
    
    const updateCommonScopeData = function (data) {
        $scope.styleSheetUrl = utils.getStylesheetUrl(data.pluginId);
    }
    
    const updateScopeData = function (data) {
              
        updateCommonScopeData(data)
        $scope.config.inputDatasetColumns = data.input_dataset_columns;
        $scope.inputDatasetColNames = data.input_dataset_columns.map(col => col.name);
                
        $scope.categoricalOptions = ["Dummy encoding", "Ordinal encoding"];
        $scope.numericOptions = ["Standard rescaling","Min-max rescaling"];
        $scope.categoricalOptions2 = ["Most frequent value", "Constant"];
        $scope.numericOptions2 = ["Average", "Median", "Constant"];
        
        $scope.updateSelectedOptions = function (columnName, optionNum) {
            const selectedOptionKey = "selectedOption" + optionNum;
            const selectedOption = $scope[selectedOptionKey][columnName];
            const isFeatureSelected = selectedOption !== "";
            $scope.showOptions[columnName] = isFeatureSelected;
        };

        $scope.toggleFeatureSelection = function (columnName) {
            const isFeatureSelected = $scope.selectedInputColumns[columnName];
            if (!isFeatureSelected) {
                $scope.showOptions[columnName] = false;
            }
        };

    };    
    
    $scope.metrics = [];

    $scope.updateMetrics = function() {
        $scope.metrics = [];

        if ($scope.config.prediction_type === 'two-class classification') {
            $scope.metrics = ['ROC AUC', 'F1 Score', 'Accuracy', 'Precision', 'Recall'];
            $scope.config.lasso_regression = false;
            $scope.config.random_forest_regression = false;
            $scope.config.xgb_regression = false;
            $scope.config.lgbm_regression = false;
            $scope.config.gb_regression = false;
            $scope.config.decision_tree_regression = false;
        } else if ($scope.config.prediction_type === 'regression') {
            $scope.metrics = ['R2', 'MAE', 'MSE', 'D2 (GLM Only)'];
            $scope.config.logistic_regression = false;
            $scope.config.random_forest_classification = false;
            $scope.config.xgb_classification = false;
            $scope.config.lgbm_classification = false;
            $scope.config.gb_classification = false;
            $scope.config.decision_tree_classification = false;
        }
    };
    
    $scope.updateMetrics();
    
    
    const init = function () {
        $scope.finishedLoading = false;   
        $scope.updateMetrics();
        utils.retrieveInfoBackend($scope, "get-info-retrain", updateScopeData);
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