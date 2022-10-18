Titanic_table = readtable('train.csv');
Titanic_data = (table2cell(Titanic_table));
head(Titanic_table)
class = cell2mat(Titanic_data(:,3));
Gender = Titanic_data(:,5);
Age = cell2mat(Titanic_data(:,6));
SibSP = cell2mat(Titanic_data(:,7));
Parch = cell2mat(Titanic_data(:,6));
survived = cell2mat(Titanic_data(:,2));
%% 
% Feature engineering and data visualization section
% 
% Feature engineering

%Feature engineering 
%Replacing Nan values in age by mean age value
mean_age = mean(Age,'omitnan');
%disp(mean_age)
Age(isnan(Age))= mean_age
%%
% Discretize age to bins
Age_Category= discretize(Age, [0 15 30 65 85],'categorical',{'Under 15', '15-30', '30-65', 'Above 65'});
Age_Category = grp2idx(Age_Category);
disp(Age_Category)
%%
Gender = grp2idx(Gender);
tbl = table(class,Gender, Age_Category, survived)
bar(Titanic_table,"Age","Survived")