class CourseCoveringProblem:
    """Course covering optimization problem solver matching exact mathematical formulation."""
    
    def __init__(self, courses: List[str], professors: List[str], terms: List[str],
                 course_preferences: Dict[Tuple[str, str], float],  # c_ij (0-10 scale)
                 term_preferences: Dict[Tuple[str, str], float],    # t_jk (0-10 scale)
                 course_streams: Dict[Tuple[str, str], int],        # n_ik: streams per course per term
                 professor_total_load: Dict[str, int],              # b_j: total courses per professor
                 professor_term_limits: Dict[Tuple[str, str], int], # L_jk: max streams per prof per term (independent from b_j)
                 course_offerings: Dict[Tuple[str, str], int]):     # O_ik: 1 if course offered in term
        
        self.courses = courses  # S
        self.professors = professors  # P
        self.terms = terms  # T
        self.course_preferences = course_preferences  # c_ij (0-10, higher is better)
        self.term_preferences = term_preferences  # t_jk (0-10, higher is better)
        self.course_streams = course_streams  # n_ik
        self.professor_total_load = professor_total_load  # b_j (total teaching load)
        self.professor_term_limits = professor_term_limits  # L_jk (independent from b_j)
        self.course_offerings = course_offerings  # O_ik (parameter, not variable)
        
        self.model = None
        self.x_vars = {}  # x_ijk
        
        # Validate preference scores are in correct range
        self._validate_preferences()
        
    def _validate_preferences(self):
        """Validate that preference scores are in 0-10 range."""
        for (course, prof), score in self.course_preferences.items():
            if not (0 <= score <= 10):
                raise ValueError(f"Course preference c_{{{course},{prof}}} = {score} must be in range [0,10]")
        
        for (prof, term), score in self.term_preferences.items():
            if not (0 <= score <= 10):
                raise ValueError(f"Term preference t_{{{prof},{term}}} = {score} must be in range [0,10]")
    
    def build_model(self):
        """Build the optimization model matching the mathematical formulation exactly."""
        self.model = pulp.LpProblem("Course_Covering_Mathematical", pulp.LpMaximize)
        
        # Decision variables: x_ijk (only where course is offered)
        self.x_vars = {}
        for course in self.courses:
            for professor in self.professors:
                for term in self.terms:
                    # Only create variables where course is actually offered (O_ik = 1)
                    if self.course_offerings.get((course, term), 0) == 1:
                        self.x_vars[(course, professor, term)] = pulp.LpVariable(
                            f"x_{course}_{professor}_{term}", cat='Binary'
                        )
        
        # Objective function: Equation (1) - Maximize c_ij + t_jk preferences
        course_pref_term = pulp.lpSum([
            self.course_preferences.get((course, professor), 0) * self.x_vars[(course, professor, term)]
            for course in self.courses
            for professor in self.professors
            for term in self.terms
            if (course, professor, term) in self.x_vars  # Only for offered courses
        ])
        
        term_pref_term = pulp.lpSum([
            self.term_preferences.get((professor, term), 0) * self.x_vars[(course, professor, term)]
            for course in self.courses
            for professor in self.professors
            for term in self.terms
            if (course, professor, term) in self.x_vars  # Only for offered courses
        ])
        
        self.model += course_pref_term + term_pref_term
        self._add_constraints()
        
    def _add_constraints(self):
        """Add all constraints matching the mathematical formulation exactly."""
        
        # Constraint (2): Stream Load Constraint (Per Term) - L_jk limits
        # sum(n_ik * x_ijk) <= L_jk for all j, k
        for professor in self.professors:
            for term in self.terms:
                term_stream_load = pulp.lpSum([
                    self.course_streams.get((course, term), 0) * self.x_vars[(course, professor, term)]
                    for course in self.courses
                    if (course, professor, term) in self.x_vars  # Only for offered courses
                ])
                # L_jk is independent constraint - professor j's limit in term k
                max_streams_in_term = self.professor_term_limits.get((professor, term), 0)
                self.model += (
                    term_stream_load <= max_streams_in_term,
                    f"StreamLoad_L_{professor}_{term}"
                )
        
        # Constraint (3): Course Load Constraint (Total) - b_j limits  
        # sum(x_ijk) <= b_j for all j (total courses across all terms)
        for professor in self.professors:
            total_courses_assigned = pulp.lpSum([
                self.x_vars[(course, professor, term)]
                for course in self.courses
                for term in self.terms
                if (course, professor, term) in self.x_vars  # Only for offered courses
            ])
            # b_j is total teaching load for professor j (independent from L_jk)
            self.model += (
                total_courses_assigned <= self.professor_total_load[professor],
                f"TotalLoad_b_{professor}"
            )
        
        # Constraint (4): Course Offering Constraint - Each offered course gets exactly one academic
        # sum(x_ijk) = O_ik for all i, k where O_ik = 1
        for course in self.courses:
            for term in self.terms:
                if self.course_offerings.get((course, term), 0) == 1:  # Only if course is offered (O_ik = 1)
                    self.model += (
                        pulp.lpSum([
                            self.x_vars[(course, professor, term)]
                            for professor in self.professors
                            if (course, professor, term) in self.x_vars
                        ]) == 1,  # Exactly one professor must be assigned
                        f"CourseOffering_O_{course}_{term}"
                    )
    
    def solve(self):
        """Solve the optimization problem."""
        if self.model is None:
            self.build_model()
        
        solver = pulp.PULP_CBC_CMD(msg=0)
        self.model.solve(solver)
        
        status = pulp.LpStatus[self.model.status]
        
        if status == 'Optimal':
            return self._extract_solution()
        else:
            return {
                'status': status,
                'objective_value': None,
                'assignments': {},
                'professor_loads': {},
                'unassigned_offerings': [],
                'constraint_violations': self._analyze_infeasibility() if status == 'Infeasible' else None
            }
    
    def _extract_solution(self):
        """Extract solution from solved model."""
        assignments = {}  # {(course, term): professor}
        professor_loads = {
            prof: {
                'total_courses': 0,  # Tracks b_j constraint
                'streams_per_term': {term: 0 for term in self.terms},  # Tracks L_jk constraints
                'total_streams': 0
            } 
            for prof in self.professors
        }
        unassigned_offerings = []
        
        # Extract assignments
        for course in self.courses:
            for term in self.terms:
                if self.course_offerings.get((course, term), 0) == 1:  # Course should be offered
                    assigned = False
                    for professor in self.professors:
                        if (course, professor, term) in self.x_vars and self.x_vars[(course, professor, term)].varValue == 1:
                            assignments[(course, term)] = professor
                            professor_loads[professor]['total_courses'] += 1  # Count for b_j
                            streams_count = self.course_streams.get((course, term), 1)
                            professor_loads[professor]['streams_per_term'][term] += streams_count  # Count for L_jk
                            professor_loads[professor]['total_streams'] += streams_count
                            assigned = True
                            break
                    
                    if not assigned:
                        unassigned_offerings.append((course, term))
        
        return {
            'status': 'Optimal',
            'objective_value': pulp.value(self.model.objective),
            'assignments': assignments,
            'professor_loads': professor_loads,
            'unassigned_offerings': unassigned_offerings
        }
    
    def _analyze_infeasibility(self):
        """Analyze potential constraint violations if problem is infeasible."""
        violations = []
        
        # Check if total required streams exceed total available capacity
        total_required_streams = sum([
            self.course_streams.get((course, term), 1)
            for course in self.courses
            for term in self.terms
            if self.course_offerings.get((course, term), 0) == 1
        ])
        
        total_available_streams = sum([
            self.professor_term_limits.get((prof, term), 0)
            for prof in self.professors
            for term in self.terms
        ])
        
        if total_required_streams > total_available_streams:
            violations.append(f"Total required streams ({total_required_streams}) exceed total available capacity ({total_available_streams})")
        
        # Check total course constraints
        total_required_courses = sum([
            1 for course in self.courses
            for term in self.terms
            if self.course_offerings.get((course, term), 0) == 1
        ])
        
        total_available_courses = sum(self.professor_total_load.values())
        
        if total_required_courses > total_available_courses:
            violations.append(f"Total required courses ({total_required_courses}) exceed total available slots ({total_available_courses})")
        
        return violations
    
    def get_model_summary(self):
        """Get summary of model parameters for validation."""
        summary = {
            'courses': len(self.courses),
            'professors': len(self.professors),
            'terms': len(self.terms),
            'total_offerings': sum(self.course_offerings.values()),
            'total_streams_required': sum([
                self.course_streams.get((course, term), 1)
                for course in self.courses
                for term in self.terms
                if self.course_offerings.get((course, term), 0) == 1
            ]),
            'total_capacity_streams': sum([
                self.professor_term_limits.get((prof, term), 0)
                for prof in self.professors
                for term in self.terms
            ]),
            'total_capacity_courses': sum(self.professor_total_load.values()),
            'preference_ranges': {
                'course_preferences': (
                    min(self.course_preferences.values()) if self.course_preferences else 0,
                    max(self.course_preferences.values()) if self.course_preferences else 0
                ),
                'term_preferences': (
                    min(self.term_preferences.values()) if self.term_preferences else 0,
                    max(self.term_preferences.values()) if self.term_preferences else 0
                )
            }
        }
        return summary
