import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pulp
import numpy as np
from typing import Dict, List, Tuple
import io

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


def main():
    st.set_page_config(
        page_title="Course Covering Optimizer - Mathematical Formulation",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 Course Covering Optimizer - Mathematical Formulation")
    st.markdown("Optimize faculty assignments using exact mathematical formulation with L_jk and b_j constraints")
    st.markdown("---")
    
    # Input method selection
    input_method = st.sidebar.selectbox(
        "Choose Input Method:",
        ["Manual Input", "Excel Upload"]
    )
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'courses' not in st.session_state:
        st.session_state.courses = []
    if 'professors' not in st.session_state:
        st.session_state.professors = []
    if 'terms' not in st.session_state:
        st.session_state.terms = ['T1', 'T2', 'T3']
    
    # Navigation based on input method
    if input_method == "Excel Upload":
        if st.session_state.step == 1:
            show_excel_upload_step()
        elif st.session_state.step == 2:
            show_results_step()
    else:  # Manual Input
        if st.session_state.step == 1:
            show_setup_step()
        elif st.session_state.step == 2:
            show_constraints_step()
        elif st.session_state.step == 3:
            show_preferences_step()
        elif st.session_state.step == 4:
            show_results_step()


def show_setup_step():
    """Show the setup step for courses and professors."""
    st.header("Step 1: Setup Courses and Professors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Courses")
        courses_input = st.text_area(
            "Enter courses (one per line):",
            value="ACTL1\nACTL2\nACTL3\nACTL4\nACTL5",
            height=120
        )
    
    with col2:
        st.subheader("Professors")
        professors_input = st.text_area(
            "Enter professors (one per line):",
            value="Jonathan\nJK\nPatrick\nAndres",
            height=120
        )
    
    # Process input and move to next step
    if st.button("Next: Set Constraints", type="primary"):
        courses = [course.strip() for course in courses_input.split('\n') if course.strip()]
        professors = [prof.strip() for prof in professors_input.split('\n') if prof.strip()]
        
        if not courses or not professors:
            st.error("Please enter at least one course and one professor.")
        else:
            st.session_state.courses = courses
            st.session_state.professors = professors
            st.session_state.step = 2
            st.rerun()


def show_constraints_step():
    """Show constraints configuration step."""
    st.header("Step 2: Configure Constraints")
    
    courses = st.session_state.courses
    professors = st.session_state.professors
    terms = st.session_state.terms
    
    # Course offerings and streams (O_ik and n_ik)
    st.subheader("Course Offerings & Streams (O_ik, n_ik)")
    st.markdown("Configure which courses are offered in which terms and how many streams each has")
    
    course_offerings = {}
    course_streams = {}
    
    for course in courses:
        st.write(f"**{course}**")
        cols = st.columns(len(terms))
        
        for idx, term in enumerate(terms):
            with cols[idx]:
                offered = st.checkbox(f"Offer in {term}", key=f"offer_{course}_{term}")
                course_offerings[(course, term)] = 1 if offered else 0
                
                if offered:
                    streams = st.number_input(
                        f"Streams in {term}:",
                        min_value=1, max_value=5, value=1,
                        key=f"streams_{course}_{term}"
                    )
                    course_streams[(course, term)] = streams
                else:
                    course_streams[(course, term)] = 0
    
    # Professor constraints (b_j and L_jk)
    st.subheader("Professor Constraints")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Total Teaching Load (b_j)**")
        st.markdown("Maximum total courses per professor across all terms")
        professor_total_load = {}
        
        for professor in professors:
            load = st.number_input(
                f"{professor} - Total courses (b_j):",
                min_value=1, max_value=10, value=4,
                key=f"total_load_{professor}"
            )
            professor_total_load[professor] = load
    
    with col2:
        st.write("**Term Stream Limits (L_jk)**")
        st.markdown("Maximum streams per professor per term (independent from b_j)")
        professor_term_limits = {}
        
        for professor in professors:
            st.write(f"**{professor}:**")
            sub_cols = st.columns(len(terms))
            
            for idx, term in enumerate(terms):
                with sub_cols[idx]:
                    limit = st.number_input(
                        f"{term}:",
                        min_value=0, max_value=6, value=2,
                        key=f"term_limit_{professor}_{term}",
                        help=f"Max streams {professor} can teach in {term}"
                    )
                    professor_term_limits[(professor, term)] = limit
    
    # Show constraint summary
    st.subheader("Constraint Summary")
    
    # Course offerings matrix
    st.write("**Course Offerings Matrix (O_ik):**")
    offerings_data = []
    for course in courses:
        row = {'Course': course}
        for term in terms:
            if course_offerings.get((course, term), 0) == 1:
                streams = course_streams.get((course, term), 1)
                row[term] = f"✓ ({streams} streams)"
            else:
                row[term] = "✗"
        offerings_data.append(row)
    
    offerings_df = pd.DataFrame(offerings_data)
    st.dataframe(offerings_df, hide_index=True)
    
    # Professor limits summary
    st.write("**Professor Constraints Summary:**")
    prof_summary = []
    for prof in professors:
        row = {'Professor': prof, 'Total Load (b_j)': professor_total_load[prof]}
        for term in terms:
            row[f"{term} Limit (L_jk)"] = professor_term_limits[(prof, term)]
        prof_summary.append(row)
    
    prof_df = pd.DataFrame(prof_summary)
    st.dataframe(prof_df, hide_index=True)
    
    # Store in session state
    st.session_state.course_offerings = course_offerings
    st.session_state.course_streams = course_streams
    st.session_state.professor_total_load = professor_total_load
    st.session_state.professor_term_limits = professor_term_limits
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Setup"):
            st.session_state.step = 1
            st.rerun()
    
    with col2:
        if st.button("Next: Set Preferences", type="primary"):
            st.session_state.step = 3
            st.rerun()


def show_preferences_step():
    """Show preferences configuration step."""
    st.header("Step 3: Set Preferences (0-10 Scale)")
    
    courses = st.session_state.courses
    professors = st.session_state.professors
    terms = st.session_state.terms
    
    # Course Preferences (c_ij)
    st.subheader("Course Preferences (c_ij)")
    st.markdown("**Scale: 0 = Cannot teach, 5 = Neutral, 10 = Strongly prefer**")
    
    course_preferences = {}
    
    for professor in professors:
        st.write(f"**{professor}'s Course Preferences:**")
        cols = st.columns(min(4, len(courses)))
        
        for idx, course in enumerate(courses):
            with cols[idx % len(cols)]:
                pref = st.slider(
                    f"{course}",
                    min_value=0, max_value=10, value=5,
                    key=f"course_pref_{course}_{professor}"
                )
                course_preferences[(course, professor)] = pref
    
    # Term Preferences (t_jk)
    st.subheader("Term Preferences (t_jk)")
    st.markdown("**Scale: 0 = Cannot teach, 5 = Neutral, 10 = Strongly prefer**")
    
    term_preferences = {}
    
    for professor in professors:
        st.write(f"**{professor}'s Term Preferences:**")
        cols = st.columns(len(terms))
        
        for idx, term in enumerate(terms):
            with cols[idx]:
                pref = st.slider(
                    f"{term}",
                    min_value=0, max_value=10, value=5,
                    key=f"term_pref_{professor}_{term}"
                )
                term_preferences[(professor, term)] = pref
    
    # Show preference matrices
    st.subheader("Preference Matrices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Course Preferences (c_ij):**")
        course_pref_matrix = pd.DataFrame(index=courses, columns=professors)
        for course in courses:
            for prof in professors:
                course_pref_matrix.loc[course, prof] = course_preferences.get((course, prof), 0)
        st.dataframe(course_pref_matrix)
    
    with col2:
        st.write("**Term Preferences (t_jk):**")
        term_pref_matrix = pd.DataFrame(index=professors, columns=terms)
        for prof in professors:
            for term in terms:
                term_pref_matrix.loc[prof, term] = term_preferences.get((prof, term), 0)
        st.dataframe(term_pref_matrix)
    
    # Store preferences
    st.session_state.course_preferences = course_preferences
    st.session_state.term_preferences = term_preferences
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Constraints"):
            st.session_state.step = 2
            st.rerun()
    
    with col2:
        if st.button("Run Optimization", type="primary"):
            st.session_state.step = 4
            st.rerun()


def show_results_step():
    """Show optimization results."""
    st.header("Step 4: Optimization Results")
    
    # Get all parameters from session state
    courses = st.session_state.courses
    professors = st.session_state.professors
    terms = st.session_state.terms
    course_offerings = st.session_state.course_offerings
    course_streams = st.session_state.course_streams
    professor_total_load = st.session_state.professor_total_load
    professor_term_limits = st.session_state.professor_term_limits
    course_preferences = st.session_state.course_preferences
    term_preferences = st.session_state.term_preferences
    
    # Run optimization
    with st.spinner("Running optimization with mathematical formulation..."):
        try:
            problem = CourseCoveringProblem(
                courses=courses,
                professors=professors,
                terms=terms,
                course_preferences=course_preferences,  # c_ij (0-10)
                term_preferences=term_preferences,      # t_jk (0-10) 
                course_streams=course_streams,          # n_ik
                professor_total_load=professor_total_load,    # b_j
                professor_term_limits=professor_term_limits,  # L_jk
                course_offerings=course_offerings       # O_ik
            )
            
            solution = problem.solve()
            
        except ValueError as e:
            st.error(f"Invalid preferences: {str(e)}")
            return
        except Exception as e:
            st.error(f"Optimization error: {str(e)}")
            return
    
    # Display results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", solution['status'])
    with col2:
        if solution['objective_value']:
            st.metric("Objective Value", f"{solution['objective_value']:.1f}")
        else:
            st.metric("Objective Value", "N/A")
    with col3:
        unassigned_count = len(solution.get('unassigned_offerings', []))
        st.metric("Unassigned Offerings", unassigned_count)
    
    if solution['status'] == 'Optimal':
        # Course assignments
        st.subheader("Course Assignments")
        
        if solution['assignments']:
            assignments_data = []
            for (course, term), professor in solution['assignments'].items():
                streams = course_streams.get((course, term), 1)
                assignments_data.append({
                    'Course': course,
                    'Term': term, 
                    'Professor': professor,
                    'Streams': streams
                })
            
            assignments_df = pd.DataFrame(assignments_data)
            st.dataframe(assignments_df, hide_index=True)
        
        # Professor workload analysis
        st.subheader("Professor Workload Analysis")
        
        workload_data = []
        for professor in professors:
            prof_load = solution['professor_loads'][professor]
            
            # b_j constraint check
            total_courses = prof_load['total_courses']
            max_total = professor_total_load[professor]
            
            workload_data.append({
                'Professor': professor,
                'Total Courses': f"{total_courses}/{max_total}",
                'Total Utilization %': f"{(total_courses/max_total)*100:.1f}%"
            })
            
            # L_jk constraint check per term
            for term in terms:
                streams_in_term = prof_load['streams_per_term'][term]
                max_streams = professor_term_limits.get((professor, term), 0)
                utilization = (streams_in_term/max_streams)*100 if max_streams > 0 else 0
                
                workload_data.append({
                    'Professor': f"  {term}",
                    'Total Courses': f"{streams_in_term}/{max_streams} streams",
                    'Total Utilization %': f"{utilization:.1f}%"
                })
        
        workload_df = pd.DataFrame(workload_data)
        st.dataframe(workload_df, hide_index=True)
        
        # Unassigned courses
        if solution.get('unassigned_offerings'):
            st.subheader("⚠️ Unassigned Course Offerings")
            for course, term in solution['unassigned_offerings']:
                streams = course_streams.get((course, term), 1)
                st.error(f"**{course}** in **{term}** ({streams} streams) - Could not assign")
        else:
            st.success("🎉 All course offerings successfully assigned!")
            
    elif solution['status'] == 'Infeasible':
        st.error("❌ Problem is infeasible - no solution exists")
        
        if solution.get('constraint_violations'):
            st.subheader("Possible Issues:")
            for violation in solution['constraint_violations']:
                st.write(f"• {violation}")
        
        st.subheader("Suggestions:")
        st.write("• Increase professor term limits (L_jk)")
        st.write("• Increase total course loads (b_j)")
        st.write("• Reduce number of course offerings")
        st.write("• Increase preference scores (avoid too many 0s)")
        
    else:
        st.error(f"Optimization failed: {solution['status']}")
    
    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Preferences"):
            st.session_state.step = 3
            st.rerun()
    
    with col2:
        if st.button("🔄 Start Over"):
            # Reset all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def show_excel_upload_step():
    """Show Excel upload interface with specific sheet structure."""
    st.header("Excel File Upload")
    
    # Show required Excel format
    st.subheader("Required Excel File Format")
    st.markdown("""
    Your Excel file must contain exactly 4 sheets with this structure:
    
    **Sheet 1: c_ij (Course Preferences)**
    - Staff names in column A starting from A3
    - Course names in row 1 starting from B1
    - Course codes in row 2 starting from B2
    - Preference scores (0-10) in the data area
    
    **Sheet 2: t_jk, L_jk, b_j (Professor Constraints)**
    - Staff names in column A starting from A3
    - t_jk (Term Preferences): 3 columns for T1, T2, T3 (0-10 scale)
    - L_jk (Term Max Load): 3 columns for T1, T2, T3 (max streams per term)
    - b_j (Total Teaching Load): 1 column for total courses
    
    **Sheet 3: O_jk (Course Offerings)**
    - Course codes in column A
    - T1, T2, T3 columns with 1/0 values (1 = offered, 0 = not offered)
    
    **Sheet 4: n_jk (Course Streams)**
    - Course codes in column A  
    - T1, T2, T3 columns with stream counts (number of streams per course per term)
    """)
    
    # Show sample format
    with st.expander("Click to see sample Excel format"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Sheet 1 - c_ij (Course Preferences):**")
            st.code("""
    A       B       C       D
1           ACTL1   ACTL2   ACTL3
2           Act1    Act2    Act3  
3   Jonathan  8       5       3
4   JK        6       9       4
5   Patrick   4       7       8
            """)
            
            st.write("**Sheet 2 - Professor Constraints:**")
            st.code("""
    A        B    C    D    E    F    G    H
1           T1   T2   T3   T1   T2   T3   Total
2           (Term Prefs) (Max Streams) Load
3   Jonathan  9    5    7    3    2    3    5
4   JK        6    8    4    2    3    2    4
5   Patrick   7    6    9    3    3    3    6
            """)
        
        with col2:
            st.write("**Sheet 3 - O_jk (Course Offerings):**")
            st.code("""
    A       B    C    D
1   Course  T1   T2   T3
2   ACTL1   1    0    1
3   ACTL2   0    1    0
4   ACTL3   1    1    1
            """)
            
            st.write("**Sheet 4 - n_jk (Course Streams):**")
            st.code("""
    A       B    C    D
1   Course  T1   T2   T3
2   ACTL1   2    0    1
3   ACTL2   0    3    0
4   ACTL3   1    2    2
            """)
    
    # Download template button
    if st.button("Download Excel Template"):
        template_data = create_excel_template_structured()
        if template_data:
            st.download_button(
                label="Download Template File",
                data=template_data,
                file_name="course_optimizer_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error("Excel template generation requires additional libraries. Please create manually using the format above.")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your Excel file:",
        type=['xlsx', 'xls'],
        help="Upload an Excel file with the 4 required sheets"
    )
    
    if uploaded_file is not None:
        try:
            # Read Excel file
            with st.spinner("Reading Excel file..."):
                excel_data = pd.ExcelFile(uploaded_file)
                
                # Validate required sheets
                required_sheets = ['c_ij', 't_jk_L_jk_b_j', 'O_jk', 'n_jk']
                available_sheets = excel_data.sheet_names
                
                st.write("Available sheets:", available_sheets)
                
                # Use flexible sheet name matching
                sheet_mapping = {}
                for req_sheet in required_sheets:
                    matched = False
                    for avail_sheet in available_sheets:
                        if req_sheet.lower() in avail_sheet.lower() or avail_sheet.lower() in req_sheet.lower():
                            sheet_mapping[req_sheet] = avail_sheet
                            matched = True
                            break
                    if not matched:
                        # Try to map by position
                        if req_sheet == 'c_ij' and len(available_sheets) > 0:
                            sheet_mapping[req_sheet] = available_sheets[0]
                        elif req_sheet == 't_jk_L_jk_b_j' and len(available_sheets) > 1:
                            sheet_mapping[req_sheet] = available_sheets[1]
                        elif req_sheet == 'O_jk' and len(available_sheets) > 2:
                            sheet_mapping[req_sheet] = available_sheets[2]
                        elif req_sheet == 'n_jk' and len(available_sheets) > 3:
                            sheet_mapping[req_sheet] = available_sheets[3]
                
                st.write("Sheet mapping:", sheet_mapping)
                
                if len(sheet_mapping) != 4:
                    st.error(f"Could not find all required sheets. Found: {list(sheet_mapping.values())}")
                    return
                
                # Read all sheets
                sheet1 = pd.read_excel(uploaded_file, sheet_name=sheet_mapping['c_ij'], header=None)
                sheet2 = pd.read_excel(uploaded_file, sheet_name=sheet_mapping['t_jk_L_jk_b_j'])
                sheet3 = pd.read_excel(uploaded_file, sheet_name=sheet_mapping['O_jk'])
                sheet4 = pd.read_excel(uploaded_file, sheet_name=sheet_mapping['n_jk'])
                
                # Process Sheet 1: c_ij (Course Preferences)
                st.subheader("Processing Course Preferences (c_ij)")
                
                # Extract course names from B1 onwards and course codes from B2 onwards
                course_names = []
                course_codes = []
                for col_idx in range(1, len(sheet1.columns)):
                    if pd.notna(sheet1.iloc[0, col_idx]):  # Row 1 (0-indexed)
                        course_names.append(str(sheet1.iloc[0, col_idx]).strip())
                    if pd.notna(sheet1.iloc[1, col_idx]):  # Row 2 (0-indexed)  
                        course_codes.append(str(sheet1.iloc[1, col_idx]).strip())
                
                # Extract staff names from A3 onwards
                staff_names = []
                for row_idx in range(2, len(sheet1)):  # Starting from row 3 (0-indexed = 2)
                    if pd.notna(sheet1.iloc[row_idx, 0]):
                        staff_names.append(str(sheet1.iloc[row_idx, 0]).strip())
                
                # Extract course preferences
                course_preferences = {}
                for staff_idx, staff in enumerate(staff_names):
                    for course_idx, course in enumerate(course_codes):
                        if course_idx < len(course_codes):
                            row_idx = staff_idx + 2  # Start from row 3
                            col_idx = course_idx + 1  # Start from column B
                            if row_idx < len(sheet1) and col_idx < len(sheet1.columns):
                                pref_val = sheet1.iloc[row_idx, col_idx]
                                if pd.notna(pref_val):
                                    course_preferences[(course, staff)] = float(pref_val)
                                else:
                                    course_preferences[(course, staff)] = 0.0
                
                st.success(f"Found {len(staff_names)} staff and {len(course_codes)} courses")
                st.write("Staff:", staff_names)
                st.write("Courses:", course_codes)
                
                # Process Sheet 2: t_jk, L_jk, b_j
                st.subheader("Processing Professor Constraints")
                
                terms = ['T1', 'T2', 'T3']
                term_preferences = {}
                professor_term_limits = {}
                professor_total_load = {}
                
                # Assuming columns are: Staff, T1_pref, T2_pref, T3_pref, T1_limit, T2_limit, T3_limit, Total_load
                for idx, row in sheet2.iterrows():
                    if pd.notna(row.iloc[0]):  # Staff name exists
                        staff = str(row.iloc[0]).strip()
                        
                        # Term preferences (columns 1-3)
                        for i, term in enumerate(terms):
                            if len(row) > i + 1 and pd.notna(row.iloc[i + 1]):
                                term_preferences[(staff, term)] = float(row.iloc[i + 1])
                            else:
                                term_preferences[(staff, term)] = 5.0  # Default neutral
                        
                        # Term limits (columns 4-6)
                        for i, term in enumerate(terms):
                            if len(row) > i + 4 and pd.notna(row.iloc[i + 4]):
                                professor_term_limits[(staff, term)] = int(row.iloc[i + 4])
                            else:
                                professor_term_limits[(staff, term)] = 2  # Default limit
                        
                        # Total load (column 7)
                        if len(row) > 7 and pd.notna(row.iloc[7]):
                            professor_total_load[staff] = int(row.iloc[7])
                        else:
                            professor_total_load[staff] = 4  # Default total load
                
                # Process Sheet 3: O_jk (Course Offerings)
                st.subheader("Processing Course Offerings")
                
                course_offerings = {}
                for idx, row in sheet3.iterrows():
                    if pd.notna(row.iloc[0]):  # Course code exists
                        course = str(row.iloc[0]).strip()
                        for i, term in enumerate(terms):
                            if len(row) > i + 1 and pd.notna(row.iloc[i + 1]):
                                course_offerings[(course, term)] = int(row.iloc[i + 1])
                            else:
                                course_offerings[(course, term)] = 0
                
                # Process Sheet 4: n_jk (Course Streams)  
                st.subheader("Processing Course Streams")
                
                course_streams = {}
                for idx, row in sheet4.iterrows():
                    if pd.notna(row.iloc[0]):  # Course code exists
                        course = str(row.iloc[0]).strip()
                        for i, term in enumerate(terms):
                            if len(row) > i + 1 and pd.notna(row.iloc[i + 1]):
                                course_streams[(course, term)] = int(row.iloc[i + 1])
                            else:
                                course_streams[(course, term)] = 0
                
                # Store in session state
                st.session_state.courses = course_codes
                st.session_state.professors = staff_names
                st.session_state.terms = terms
                st.session_state.course_preferences = course_preferences
                st.session_state.term_preferences = term_preferences
                st.session_state.professor_term_limits = professor_term_limits
                st.session_state.professor_total_load = professor_total_load
                st.session_state.course_offerings = course_offerings
                st.session_state.course_streams = course_streams
                
                st.success("Excel file loaded successfully!")
                
                # Show summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Courses", len(course_codes))
                with col2:
                    st.metric("Professors", len(staff_names))
                with col3:
                    offerings_count = sum(course_offerings.values())
                    st.metric("Course Offerings", offerings_count)
                
                # Show loaded data preview
                st.subheader("Data Preview")
                
                # Course preferences preview
                with st.expander("Course Preferences (c_ij)"):
                    if course_preferences:
                        pref_matrix = pd.DataFrame(index=course_codes, columns=staff_names)
                        for course in course_codes:
                            for staff in staff_names:
                                pref_matrix.loc[course, staff] = course_preferences.get((course, staff), 0)
                        st.dataframe(pref_matrix)
                
                # Professor constraints preview
                with st.expander("Professor Constraints"):
                    constraints_data = []
                    for staff in staff_names:
                        row_data = {'Professor': staff, 'Total Load (b_j)': professor_total_load.get(staff, 0)}
                        for term in terms:
                            row_data[f'{term} Pref (t_jk)'] = term_preferences.get((staff, term), 0)
                            row_data[f'{term} Limit (L_jk)'] = professor_term_limits.get((staff, term), 0)
                        constraints_data.append(row_data)
                    
                    constraints_df = pd.DataFrame(constraints_data)
                    st.dataframe(constraints_df)
                
                # Course offerings and streams preview
                with st.expander("Course Offerings & Streams"):
                    offerings_data = []
                    for course in course_codes:
                        row_data = {'Course': course}
                        for term in terms:
                            offering = course_offerings.get((course, term), 0)
                            streams = course_streams.get((course, term), 0)
                            if offering == 1:
                                row_data[term] = f"✓ ({streams} streams)"
                            else:
                                row_data[term] = "✗"
                        offerings_data.append(row_data)
                    
                    offerings_df = pd.DataFrame(offerings_data)
                    st.dataframe(offerings_df)
                
                # Run optimization button
                if st.button("Run Optimization", type="primary"):
                    st.session_state.step = 2
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            st.write("Please check that your file format matches the required template.")
            
            # Show the actual error for debugging
            st.code(str(e))


def create_excel_template_structured():
    """Create an Excel template file with the specified structure."""
    # This would require xlsxwriter or openpyxl
    # For now, return None to indicate manual creation needed
    return None
