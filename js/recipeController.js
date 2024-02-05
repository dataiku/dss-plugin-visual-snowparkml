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
        $scope.inputDatasetColumns = data.input_dataset_columns;
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
        } else if ($scope.config.prediction_type === 'regression') {
            $scope.metrics = ['R2', 'MAE', 'MSE', 'D2 (GLM Only)'];
            $scope.config.logistic_regression = false;
            $scope.config.random_forest_classification = false;
        }
    };
    
    $scope.updateMetrics();
    
    
    const init = function () {
        $scope.finishedLoading = false;   
        $scope.updateMetrics();
        utils.retrieveInfoBackend($scope, "get-info-retrain", updateScopeData);
        utils.initVariable($scope, 'col_label', '');
        utils.initVariable($scope, 'time_ordering_variable', '');
    };

    init();
    console.log($scope.config.col_label);
    console.log($scope.config.time_ordering_variable);
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
        return `/plugins/${pluginId}/resource/stylesheets/dl-image-toolbox.css`;
    };
});