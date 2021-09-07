function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
num_data_points = size(X,1);
for current_row = 1:num_data_points
	data_point = X(current_row,:);
	best = 10000000;
	best_cent = -1;
	for k = 1:K
		current_centroid = centroids(k,:);
		dist_vec = (current_centroid - data_point).^2;
		dist = sum(sum(dist_vec));
		if dist < best
			best_cent = k;
			best = dist;
		endif
	endfor
	idx(current_row) = best_cent;
endfor





% =============================================================

end

