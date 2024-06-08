# This file was generated, do not modify it. # hide
missing_data = similar(my_data, Missing) # vector of `missing`
model_missing = dice_throw(missing_data) # instantiate the "predictive model
prior_check = predict(model_missing, prior_chain);